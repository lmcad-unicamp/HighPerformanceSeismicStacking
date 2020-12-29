#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/traveltime/CommonMidPoint.hpp"
#include "common/include/traveltime/CommonReflectionSurface.hpp"
#include "common/include/traveltime/OffsetContinuationTrajectory.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/algorithm/CudaLinearSearchAlgorithm.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"
#include "cuda/include/semblance/kernel/cmp/common.cuh"
#include "cuda/include/semblance/kernel/cmp/linear_search.cuh"
#include "cuda/include/semblance/kernel/zocrs/common.cuh"
#include "cuda/include/semblance/kernel/zocrs/linear_search.cuh"
#include "cuda/include/semblance/kernel/oct/linear_search.cuh"

#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

#ifdef PROFILE_ENABLED
#include <cuda_profiler_api.h>
#endif

using namespace std;

CudaLinearSearchAlgorithm::CudaLinearSearchAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : LinearSearchAlgorithm(traveltime, context, dataBuilder) {
}

void CudaLinearSearchAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {

    LOGI("Computing semblance for m0 = " << m0);

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

    dim3 dimGrid(static_cast<int>(ceil(static_cast<float>(totalNumberOfParameters * samplesPerTrace) / static_cast<float>(threadCount))));

#ifdef PROFILE_ENABLED
    LOGI("CUDA profiler is enabled.")
    CUDA_ASSERT(cudaProfilerStart());
#endif

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
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])
            );

            dim3 dimGridBest(static_cast<int>(ceil(static_cast<float>(samplesPerTrace) / static_cast<float>(threadCount))));

            selectBestSemblancesForCommonMidPoint<<< dimGridBest, threadCount >>>(
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK]),
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                samplesPerTrace,
                CUDA_DEV_PTR(deviceResultArray)
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
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])
            );

            dim3 dimGridBest(static_cast<int>(ceil(static_cast<float>(samplesPerTrace) / static_cast<float>(threadCount))));

            selectBestSemblancesForZeroOffsetCommonReflectionSurface<<< dimGridBest, threadCount >>>(
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK]),
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                samplesPerTrace,
                CUDA_DEV_PTR(deviceResultArray)
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
                gather->getApm(),
                m0,
                traveltime->getReferenceHalfoffset(),
                dtInSeconds,
                tauIndexDisplacement,
                windowSize,
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                CUDA_DEV_PTR(deviceNotUsedCountArray),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])
            );

            dim3 dimGridBest(static_cast<int>(ceil(static_cast<float>(samplesPerTrace) / static_cast<float>(threadCount))));

            selectBestSemblancesForOffsetContinuationTrajectory<<< dimGridBest, threadCount >>>(
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]),
                CUDA_DEV_PTR(commonResultDeviceArrayMap[SemblanceCommonResult::STACK]),
                CUDA_DEV_PTR(deviceParameterArray),
                totalNumberOfParameters,
                samplesPerTrace,
                CUDA_DEV_PTR(deviceResultArray)
            );

            break;
        }
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaGetLastError());

#ifdef PROFILE_ENABLED
    CUDA_ASSERT(cudaProfilerStop());
    LOGI("CUDA profiler is disabled.");
#endif
}

void CudaLinearSearchAlgorithm::initializeParameters() {

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    dim3 dimGrid(static_cast<int>(ceil(static_cast<float>(totalNumberOfParameters) / static_cast<float>(threadCount))));

    switch (traveltime->getModel()) {
        case CMP: {
            float minVelocity = traveltime->getLowerBoundForParameter(CommonMidPoint::VELOCITY);
            float increment = discretizationStep[CommonMidPoint::VELOCITY];

            buildParameterArrayForCommonMidPoint<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceParameterArray),
                minVelocity,
                increment,
                totalNumberOfParameters
            );

            break;
        }
        case ZOCRS: {
            buildParameterArrayForZeroOffsetCommonReflectionSurface<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceParameterArray),
                traveltime->getLowerBoundForParameter(CommonReflectionSurface::VELOCITY),
                discretizationStep[CommonReflectionSurface::VELOCITY],
                discretizationGranularity[CommonReflectionSurface::VELOCITY],
                traveltime->getLowerBoundForParameter(CommonReflectionSurface::A),
                discretizationStep[CommonReflectionSurface::A],
                discretizationGranularity[CommonReflectionSurface::A],
                traveltime->getLowerBoundForParameter(CommonReflectionSurface::B),
                discretizationStep[CommonReflectionSurface::B],
                discretizationGranularity[CommonReflectionSurface::B],
                totalNumberOfParameters
            );

            break;
        }
        case OCT: {
            buildParameterArrayForOffsetContinuationTrajectory<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceParameterArray),
                traveltime->getLowerBoundForParameter(OffsetContinuationTrajectory::VELOCITY),
                discretizationStep[OffsetContinuationTrajectory::VELOCITY],
                discretizationGranularity[OffsetContinuationTrajectory::VELOCITY],
                traveltime->getLowerBoundForParameter(OffsetContinuationTrajectory::SLOPE),
                discretizationStep[OffsetContinuationTrajectory::SLOPE],
                discretizationGranularity[OffsetContinuationTrajectory::SLOPE],
                totalNumberOfParameters
            );

            break;
        }
        default:
           throw invalid_argument("Invalid traveltime model");
    }

    CUDA_ASSERT(cudaGetLastError());

    CUDA_ASSERT(cudaDeviceSynchronize());
}

void CudaLinearSearchAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

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
        case OCT: {
            unsigned int samplePop = 1024;
            vector<float> parameterSampleArray(samplePop * 2);

            default_random_engine generator;
    
            for (unsigned int prmtr = 0; prmtr < 2; prmtr++) {
    
                float min = traveltime->getLowerBoundForParameter(prmtr);
                float max = traveltime->getUpperBoundForParameter(prmtr);
    
                uniform_real_distribution<float> uniformDist(min, max);
    
                for (unsigned int idx = 0; idx < samplePop; idx++) {
                    float randomParameter = uniformDist(generator);
                    if (prmtr == OffsetContinuationTrajectory::VELOCITY) {
                        randomParameter = 4.0f / (randomParameter * randomParameter);
                    }
                    parameterSampleArray[idx * 2 + prmtr] = randomParameter;
                }
            }
    
            unique_ptr<DataContainer> selectionParameterArray(dataFactory->build(2 * samplePop, deviceContext));

            selectionParameterArray->copyFrom(parameterSampleArray);

            selectTracesForOffsetContinuationTrajectory<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::HLFOFFST]),
                CUDA_DEV_PTR(selectionParameterArray),
                samplePop,
                traceCount,
                samplesPerTrace,
                dtInSeconds,
                apm,
                m0,
                traveltime->getReferenceHalfoffset(),
                deviceUsedTraceMaskArray
            );
            break;
        }
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    CUDA_ASSERT(cudaGetLastError());

    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaMemcpy(usedTraceMask.data(), deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CUDA_ASSERT(cudaFree(deviceUsedTraceMaskArray));

    MEASURE_EXEC_TIME(copyTime, copyOnlySelectedTracesToDevice(usedTraceMask));

    LOGI("Execution time for copying traces is " << copyTime.count() << "s");
}
