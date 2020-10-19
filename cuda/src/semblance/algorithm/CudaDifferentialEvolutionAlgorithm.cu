#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/algorithm/CudaDifferentialEvolutionAlgorithm.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"
#include "cuda/include/semblance/kernel/base.h"
#include "cuda/include/semblance/kernel/differential_evolution.h"
#include "cuda/include/semblance/kernel/cmp/common.cuh"
#include "cuda/include/semblance/kernel/cmp/differential_evolution.cuh"
#include "cuda/include/semblance/kernel/zocrs/common.cuh"
#include "cuda/include/semblance/kernel/zocrs/differential_evolution.cuh"
#include "cuda/include/semblance/kernel/oct/differential_evolution.cuh"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <time.h>

using namespace std;

CudaDifferentialEvolutionAlgorithm::CudaDifferentialEvolutionAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int gen,
    unsigned int ind
) : DifferentialEvolutionAlgorithm(model, context, dataBuilder, gen, ind) {
}

CudaDifferentialEvolutionAlgorithm::~CudaDifferentialEvolutionAlgorithm() {
    CUDA_ASSERT(cudaFree(st));
}

void CudaDifferentialEvolutionAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {
    
    if (!filteredTracesCount) {
        LOGI("No trace has been selected for m0 = " << m0 << ". Skipping.");
        return;
    }

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();
    unsigned int numberOfCommonResults = traveltime->getNumberOfCommonResults();

    Gather* gather = Gather::getInstance();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    int tauIndexDisplacement = gather->getTauIndexDisplacement();
    unsigned int windowSize = gather->getWindowSize();

    dim3 dimGrid(static_cast<int>(ceil(static_cast<float>(individualsPerPopulation * samplesPerTrace) / static_cast<float>(threadCount))));

    switch (traveltime->getModel()) {
        case CMP:
            computeSemblancesForCommonMidPoint<<< dimGrid, threadCount >>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ]),
                filteredTracesCount,
                samplesPerTrace,
                individualsPerPopulation,
                dtInSeconds,
                tauIndexDisplacement,
                windowSize,
                totalNumberOfParameters,
                CUDA_DEV_PTR(deviceParameterArray),
                CUDA_DEV_PTR(deviceResultArray)
            );
            break;
        case ZOCRS:
        case OCT:
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaGetLastError());
}

void CudaDifferentialEvolutionAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

    LOGI("Selecting traces for m0 = " << m0);

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();
    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    int tauIndexDisplacement = gather->getTauIndexDisplacement();
    unsigned int windowSize = gather->getWindowSize();
    float apm = gather->getApm();    
    float h0 = traveltime->getReferenceHalfoffset();

    vector<unsigned char> usedTraceMask(traceCount);

    unsigned char* deviceUsedTraceMaskArray;
    CUDA_ASSERT(cudaMalloc((void **) &deviceUsedTraceMaskArray, traceCount * sizeof(char)));
    CUDA_ASSERT(cudaMemset(deviceUsedTraceMaskArray, 0, traceCount * sizeof(unsigned char)))

    dim3 dimGrid(static_cast<int>(ceil(static_cast<float>(traceCount) / static_cast<float>(threadCount))));

    chrono::duration<double> copyTime = chrono::duration<double>::zero();

    switch (traveltime->getModel()) {
        case CMP:
            selectTracesForCommonMidPoint<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                traceCount,
                deviceUsedTraceMaskArray,
                m0
            );
            break;
        case ZOCRS:
            selectTracesForZeroOffsetCommonReflectionSurface<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                traceCount,
                deviceUsedTraceMaskArray,
                m0,
                apm
            );
            break;
        case OCT:
            selectTracesForOffsetContinuationTrajectoryAndDifferentialEvolution<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::HLFOFFST]),
                CUDA_DEV_PTR(deviceParameterArray),
                traceCount,
                samplesPerTrace,
                individualsPerPopulation,
                dtInSeconds,
                apm,
                m0,
                h0,
                deviceUsedTraceMaskArray
            );
            break;
    }

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaMemcpy(usedTraceMask.data(), deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaFree(deviceUsedTraceMaskArray));

    MEASURE_EXEC_TIME(copyTime, copyOnlySelectedTracesToDevice(usedTraceMask));

    LOGI("Execution time for copying traces is " << copyTime.count() << "s");
}

void CudaDifferentialEvolutionAlgorithm::setupRandomSeedArray() {

    Gather* gather = Gather::getInstance();

    deviceContext->activate();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();

    dim3 dimGrid(static_cast<int>(ceil(static_cast<float>(samplesPerTrace * individualsPerPopulation) / static_cast<float>(threadCount))));

    CUDA_ASSERT(cudaMalloc(&st, samplesPerTrace * individualsPerPopulation * sizeof(curandState)));

    srand(static_cast<unsigned int>(time(NULL)));

    setupRandomSeed<<< dimGrid, threadCount >>>(st, rand());

    CUDA_ASSERT(cudaGetLastError());

    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::startAllPopulations() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    kernelStartPopulations<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(min),
        CUDA_DEV_PTR(max),
        st,
        numberOfParameters
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    fx->reset();
    fu->reset();
}

void CudaDifferentialEvolutionAlgorithm::mutateAllPopulations() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    kernelmutateAllPopulations<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(v),
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(min),
        CUDA_DEV_PTR(max),
        st,
        numberOfParameters
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::crossoverPopulationIndividuals() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    kernelCrossoverPopulationIndividuals<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(u),
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(v),
        st,
        numberOfParameters
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::advanceGeneration() {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfCommonResults = traveltime->getNumberOfCommonResults();

    kernelAdvanceGeneration<<< dimGrid, individualsPerPopulation >>>(
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(fx),
        CUDA_DEV_PTR(u),
        CUDA_DEV_PTR(fu),
        numberOfParameters,
        numberOfCommonResults
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaDifferentialEvolutionAlgorithm::selectBestIndividuals(vector<float>& resultArrays) {

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    unsigned int numberOfResults = traveltime->getNumberOfResults();
    unsigned int sharedMemCount = numberOfResults * individualsPerPopulation * static_cast<unsigned int>(sizeof(float));

    deviceResultArray.reset(dataFactory->build(numberOfResults * gather->getSamplesPerTrace(), deviceContext));

    kernelSelectBestIndividuals<<< dimGrid, individualsPerPopulation, sharedMemCount >>>(
        CUDA_DEV_PTR(x),
        CUDA_DEV_PTR(fx),
        CUDA_DEV_PTR(deviceResultArray),
        traveltime->getNumberOfParameters(),
        traveltime->getNumberOfCommonResults(),
        gather->getSamplesPerTrace()
    );

    CUDA_ASSERT(cudaGetLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());

    deviceResultArray->pasteTo(resultArrays);
}
