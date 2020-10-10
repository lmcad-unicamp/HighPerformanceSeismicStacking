#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/algorithm/CudaLinearSearchAlgorithm.hpp"
#include "cuda/include/semblance/data/CudaDataContainer.hpp"
#include "cuda/include/semblance/kernel/base.h"
#include "cuda/include/semblance/kernel/linear_search.h"

#include <cmath>
#include <numeric>
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

    Gather* gather = Gather::getInstance();

    dim3 dimGrid(gather->getSamplesPerTrace());

    gpu_gather_data_t kernelData = gather->getGpuGatherData();

    gpu_traveltime_data_t kernelTraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    unsigned int sharedMemSizeCount =
        traveltime->getNumberOfResults() * threadCount * static_cast<unsigned int>(sizeof(float));

    LOGD("threadCount = " << threadCount);
    LOGD("getNumberOfResults = " << traveltime->getNumberOfResults());
    LOGD("sharedMemSizeCount = " << sharedMemSizeCount);

#ifdef PROFILE_ENABLED
    if (filteredTracesCount) {
        LOGI("CUDA profiler is enabled.")
        CUDA_ASSERT(cudaProfilerStart());
    }
#endif

    kernelLinearSearch<<< dimGrid, threadCount, sharedMemSizeCount >>>(
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]),
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]),
        CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]),
        filteredTracesCount,
        kernelData,
        kernelReferencePoint,
        kernelTraveltime,
        CUDA_DEV_PTR(deviceParameterArray),
        CUDA_DEV_PTR(deviceResultArray),
        CUDA_DEV_PTR(deviceNotUsedCountArray)
    );

    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaGetLastError());

#ifdef PROFILE_ENABLED
    if (filteredTracesCount) {
        CUDA_ASSERT(cudaProfilerStop());
        LOGI("CUDA profiler is disabled.");
    }
#endif
}

void CudaLinearSearchAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    vector<unsigned char> usedTraceMask(traceCount);

    unsigned char* deviceUsedTraceMaskArray;
    CUDA_ASSERT(cudaMalloc((void **) &deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char)));
    CUDA_ASSERT(cudaMemset(deviceUsedTraceMaskArray, 0, traceCount * sizeof(unsigned char)))

    dim3 dimGrid(static_cast<int>(ceil(traceCount / threadCount)));

    LOGI("Using " << dimGrid.x << " blocks for traces filtering (threadCount = "<< threadCount << ")");

    gpu_gather_data_t gatherData = gather->getGpuGatherData();

    gpu_traveltime_data_t kernelTraveltime = traveltime->toGpuData();

    gpu_reference_point_t kernelReferencePoint;
    kernelReferencePoint.m0 = m0;
    kernelReferencePoint.h0 = traveltime->getReferenceHalfoffset();

    chrono::duration<double> kernelExecutionTime = chrono::duration<double>::zero();
    chrono::duration<double> copyTime = chrono::duration<double>::zero();

    switch (traveltime->getModel()) {
        case CMP:
        case ZOCRS:
            LOGD("Executing filterMidpointDependentTraces<<<>>> kernel");
            MEASURE_EXEC_TIME(kernelExecutionTime, (filterMidpointDependentTraces<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                traceCount,
                deviceUsedTraceMaskArray,
                traveltime->toGpuData(),
                gather->getApm(),
                m0
            )));
            break;

        case OCT:
            LOGD("Executing filterOutTracesForOffsetContinuationTrajectoryAndLinearSearch<<<>>> kernel");
            MEASURE_EXEC_TIME(kernelExecutionTime, (filterOutTracesForOffsetContinuationTrajectoryAndLinearSearch<<<dimGrid, threadCount>>>(
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::MDPNT]),
                CUDA_DEV_PTR(deviceFilteredTracesDataMap[GatherData::HLFOFFST]),
                deviceUsedTraceMaskArray,
                traceCount,
                gatherData,
                kernelReferencePoint,
                kernelTraveltime,
                CUDA_DEV_PTR(deviceParameterArray),
                getParameterArrayStep()
            )));
            break;
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    LOGI("Execution time for filtering kernel is " << kernelExecutionTime.count() << "s");

    CUDA_ASSERT(cudaGetLastError());

    CUDA_ASSERT(cudaDeviceSynchronize());

    CUDA_ASSERT(cudaMemcpy(usedTraceMask.data(), deviceUsedTraceMaskArray, traceCount * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CUDA_ASSERT(cudaFree(deviceUsedTraceMaskArray));

    MEASURE_EXEC_TIME(copyTime, copyOnlySelectedTracesToDevice(usedTraceMask));

    LOGI("Execution time for copying traces is " << copyTime.count() << "s");
}
