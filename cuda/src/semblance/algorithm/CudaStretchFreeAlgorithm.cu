#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"
#include "cuda/include/semblance/algorithm/CudaStretchFreeAlgorithm.hpp"
#include "cuda/include/semblance/kernel/common/stretch_free.cuh"
#include "cuda/include/semblance/kernel/cmp/common.cuh"
#include "cuda/include/semblance/kernel/cmp/stretch_free.cuh"
#include "cuda/include/semblance/kernel/zocrs/common.cuh"
#include "cuda/include/semblance/kernel/zocrs/stretch_free.cuh"
#include "cuda/include/semblance/kernel/oct/stretch_free.cuh"

#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace std;

CudaStretchFreeAlgorithm::CudaStretchFreeAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int threadCount,
    const vector<string>& files
) : StretchFreeAlgorithm(traveltime, context, dataBuilder, threadCount, files) {
}

void CudaStretchFreeAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {

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
    float apm = gather->getApm();
    float h0 = traveltime->getReferenceHalfoffset();

    dim3 dimGrid(static_cast<int>(ceil(static_cast<float>(totalNumberOfParameters * samplesPerTrace) / static_cast<float>(threadCount))));

    switch (traveltime->getModel()) {
        case CMP: {
            computeSemblancesForCommonMidPoint<<< dimGrid, threadCount >>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ]),
                filteredTracesCount,
                samplesPerTrace,
                dtInSeconds,
                tauIndexDisplacement,
                windowSize,
                CUDA_DEV_PTR(nonStretchFreeParameters[m0]),
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])
            );
            break;
        }
        case ZOCRS: {
            computeSemblancesForZeroOffsetCommonReflectionSurface<<< dimGrid, threadCount >>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ]),
                filteredTracesCount,
                samplesPerTrace,
                m0,
                dtInSeconds,
                tauIndexDisplacement,
                windowSize,
                CUDA_DEV_PTR(nonStretchFreeParameters[m0]),
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])
            );
            break;
        }
        case OCT: {
            computeSemblancesForOffsetContinuationTrajectory<<< dimGrid, threadCount >>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]),
                filteredTracesCount,
                samplesPerTrace,
                apm,
                m0,
                h0,
                dtInSeconds,
                tauIndexDisplacement,
                windowSize,
                /* Parameter arrays */
                CUDA_DEV_PTR(nonStretchFreeParameters[m0]),
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                /* Output arrays */
                CUDA_DEV_PTR(deviceNotUsedCountArray),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])
            );
        }
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    dim3 dimGridBest(static_cast<int>(ceil(static_cast<float>(samplesPerTrace) / static_cast<float>(threadCount))));

    selectBestSemblances<<< dimGridBest, threadCount >>>(
        CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
        CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK]),
        CUDA_DEV_PTR(deviceParameterArray),
        totalNumberOfParameters,
        samplesPerTrace,
        CUDA_DEV_PTR(deviceResultArray)
    );

    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetLastError());
}

void CudaStretchFreeAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();
    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    int tauIndexDisplacement = gather->getTauIndexDisplacement();
    unsigned int windowSize = gather->getWindowSize();
    float apm = gather->getApm();

    vector<unsigned char> usedTraceMask(traceCount);

    unsigned char* deviceUsedTraceMaskArray;
    CUDA_ASSERT(cudaMalloc((void **) &deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char)));
    CUDA_ASSERT(cudaMemset(deviceUsedTraceMaskArray, 0, traceCount * sizeof(unsigned char)))

    dim3 dimGrid(static_cast<int>(ceil(static_cast<float>(traceCount) / static_cast<float>(threadCount))));

    LOGI("Using " << dimGrid.x << " blocks for traces filtering (threadCount = "<< threadCount << ")");

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
            selectTracesForOffsetContinuationTrajectory<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::HLFOFFST]),
                CUDA_DEV_PTR(nonStretchFreeParameters[m0]),
                traceCount,
                samplesPerTrace,
                dtInSeconds,
                apm,
                m0,
                traveltime->getReferenceHalfoffset(),
                deviceUsedTraceMaskArray
            );
            break;
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaGetLastError());

    CUDA_ASSERT(cudaMemcpy(usedTraceMask.data(), deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CUDA_ASSERT(cudaFree(deviceUsedTraceMaskArray));

    MEASURE_EXEC_TIME(copyTime, copyOnlySelectedTracesToDevice(usedTraceMask));

    LOGD("Execution time for copying traces is " << copyTime.count() << "s");
}
