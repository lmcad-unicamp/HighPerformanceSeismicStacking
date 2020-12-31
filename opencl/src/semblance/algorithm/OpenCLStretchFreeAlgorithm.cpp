#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLStretchFreeAlgorithm.hpp"
#include "opencl/include/semblance/data/OpenCLDataContainer.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

OpenCLStretchFreeAlgorithm::OpenCLStretchFreeAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int threadCount,
    const vector<string>& files
) : StretchFreeAlgorithm(traveltime, context, dataBuilder, threadCount, files) {
}

void OpenCLStretchFreeAlgorithm::compileKernels(const string& deviceKernelSourcePath) {
    OpenCLComputeAlgorithm::compileKernels(deviceKernelSourcePath, "stretch_free", traveltime, deviceContext);
}

void OpenCLStretchFreeAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {

    if (!filteredTracesCount) {
        LOGI("No trace has been selected for m0 = " << m0 << ". Skipping.");
        return;
    }

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    Gather* gather = Gather::getInstance();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    int tauIndexDisplacement = gather->getTauIndexDisplacement();
    unsigned int windowSize = gather->getWindowSize();
    float apm = gather->getApm();
    float h0 = traveltime->getReferenceHalfoffset();

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(totalNumberOfParameters * samplesPerTrace, threadCount));
    cl::NDRange local(threadCount);

    cl_uint argIndex = 0;

    switch (traveltime->getModel()) {
        case CMP: {
            cl::Kernel& computeSemblancesForCommonMidPoint = kernels["computeSemblancesForCommonMidPoint"];

            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &filteredTracesCount));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(float), &dtInSeconds));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(int), &tauIndexDisplacement));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(int), &windowSize));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(nonStretchFreeParameters[m0])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForCommonMidPoint, offset, global, local));
            break;
        }
        case ZOCRS: {
            cl::Kernel& computeSemblancesForZeroOffsetCommonReflectionSurface = kernels["computeSemblancesForZeroOffsetCommonReflectionSurface"];

            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL])));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_MDPNT])));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ])));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &filteredTracesCount));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &m0));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &dtInSeconds));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(int), &tauIndexDisplacement));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(int), &windowSize));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(nonStretchFreeParameters[m0])));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForZeroOffsetCommonReflectionSurface, offset, global, local));
            break;
        }
        case OCT: {
            cl::Kernel& computeSemblancesForOffsetContinuationTrajectory = kernels["computeSemblancesForOffsetContinuationTrajectory"];

            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL])));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_MDPNT])));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST])));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &filteredTracesCount));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &apm));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &m0));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &h0));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &dtInSeconds));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(int), &tauIndexDisplacement));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(int), &windowSize));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(nonStretchFreeParameters[m0])));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceNotUsedCountArray)));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForOffsetContinuationTrajectory, offset, global, local));
            break;
        }
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    argIndex = 0;
    cl::Kernel& selectBestSemblances = kernels["selectBestSemblances"];

    OPENCL_ASSERT(selectBestSemblances.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
    OPENCL_ASSERT(selectBestSemblances.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));
    OPENCL_ASSERT(selectBestSemblances.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
    OPENCL_ASSERT(selectBestSemblances.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
    OPENCL_ASSERT(selectBestSemblances.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
    OPENCL_ASSERT(selectBestSemblances.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));

    cl::NDRange globalSelection(fitGlobal(samplesPerTrace, threadCount));

    OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectBestSemblances, offset, globalSelection, local));

    OPENCL_ASSERT(commandQueue.finish());
}

void OpenCLStretchFreeAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {
    cl_int errorCode;

    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int traceCount = gather->getTotalTracesCount();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    float apm = gather->getApm();
    float h0 = traveltime->getReferenceHalfoffset();

    vector<unsigned char> usedTraceMask(traceCount);

    unique_ptr<cl::Buffer> deviceUsedTraceMaskArray = make_unique<cl::Buffer>(
        openClContext->getContext(),
        CL_MEM_READ_ONLY,
        traceCount * sizeof(unsigned char),
        nullptr,
        &errorCode
    );

    OPENCL_ASSERT_CODE(errorCode);

    OPENCL_ASSERT(commandQueue.enqueueFillBuffer(
        *deviceUsedTraceMaskArray,
        0,
        0,
        traceCount * sizeof(unsigned char)
    ));

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(traceCount, threadCount));
    cl::NDRange local(threadCount);

    chrono::duration<double> copyTime = chrono::duration<double>::zero();

    switch (traveltime->getModel()) {
        case CMP: {
            cl_uint argIndex = 0;
            cl::Kernel& selectTracesForCommonMidPoint = kernels["selectTracesForCommonMidPoint"];

            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::MDPNT])));
            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &traceCount));
            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, *deviceUsedTraceMaskArray));
            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, sizeof(float), &m0));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectTracesForCommonMidPoint, offset, global, local));
            break;
        }
        case ZOCRS: {
            cl_uint argIndex = 0;
            cl::Kernel& selectTracesForZeroOffsetCommonReflectionSurface = kernels["selectTracesForZeroOffsetCommonReflectionSurface"];

            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::MDPNT])));
            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &traceCount));
            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, *deviceUsedTraceMaskArray));
            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &m0));
            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &apm));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectTracesForZeroOffsetCommonReflectionSurface, offset, global, local));
            break;
        }
        case OCT: {
            cl_uint argIndex = 0;
            cl::Kernel& selectTracesForOffsetContinuationTrajectory = kernels["selectTracesForOffsetContinuationTrajectory"];

            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::MDPNT])));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::HLFOFFST])));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(nonStretchFreeParameters[m0])));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &traceCount));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &dtInSeconds));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &apm));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &m0));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &h0));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, *deviceUsedTraceMaskArray));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectTracesForOffsetContinuationTrajectory, offset, global, local));
            break;
        }
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    OPENCL_ASSERT(commandQueue.finish());

    OPENCL_ASSERT(commandQueue.enqueueReadBuffer(
        *deviceUsedTraceMaskArray.get(),
        CL_TRUE,
        0,
        usedTraceMask.size() * sizeof(unsigned char),
        usedTraceMask.data()
    ));

    MEASURE_EXEC_TIME(copyTime, copyOnlySelectedTracesToDevice(usedTraceMask));

    LOGD("Execution time for copying traces is " << copyTime.count() << "s");
}
