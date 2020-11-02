#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/traveltime/CommonMidPoint.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLLinearSearchAlgorithm.hpp"
#include "opencl/include/semblance/data/OpenCLDataContainer.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <filesystem>
#include <cmath>
#include <memory>

using namespace std;

OpenCLLinearSearchAlgorithm::OpenCLLinearSearchAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : LinearSearchAlgorithm(traveltime, context, dataBuilder) {
}

void OpenCLLinearSearchAlgorithm::compileKernels(const string& deviceKernelSourcePath) {
    cl_int errorCode;

    vector<string> kernelSources;
    vector<cl::Kernel> openClKernels;

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    const string traveltimeKernelPath = deviceKernelSourcePath + traveltime->getTraveltimeWord();

    filesystem::path specificKernelPath(traveltimeKernelPath + "/linear_search.cl");
    filesystem::path commonKernelPath(traveltimeKernelPath + "/common.cl");

    if (!filesystem::exists(specificKernelPath)) {
        throw runtime_error("Kernels for greedy algorithm do not exist.");
    }

    kernelSources.push_back(static_cast<string>(specificKernelPath));

    if (filesystem::exists(commonKernelPath)) {
        kernelSources.push_back(static_cast<string>(commonKernelPath));
    }

    cl::Program program(openClContext->getContext(), kernelSources, &errorCode);

    OPENCL_ASSERT_CODE(errorCode);

    errorCode = program.build();

    if (errorCode != CL_SUCCESS) {
        auto errors = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openClContext->getDevice(), &errorCode);
        ostringstream stringStream;
        stringStream << "Building cl::Program failed with " << errors;
        throw runtime_error(stringStream.str());
    }

    OPENCL_ASSERT(program.createKernels(&openClKernels));

    for(auto kernel : openClKernels) {
        string kernelName = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&errorCode);
        OPENCL_ASSERT_CODE(errorCode);
        kernels[kernelName] = kernel;
    }
}

void OpenCLLinearSearchAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {

}

void OpenCLLinearSearchAlgorithm::initializeParameters() {

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    cl::NDRange offset;
    cl::NDRange global(static_cast<int>(ceil(static_cast<float>(totalNumberOfParameters) / static_cast<float>(threadCount))));
    cl::NDRange local(threadCount);

    auto commandQueue = openClContext->getCommandQueue();

    switch (traveltime->getModel()) {
        case CMP: {
            float minVelocity = traveltime->getLowerBoundForParameter(CommonMidPoint::VELOCITY);
            float increment = discretizationStep[CommonMidPoint::VELOCITY];
            cl_uint argIndex = 0;
            cl::Kernel& buildParameterArrayForCommonMidPoint = kernels["buildParameterArrayForCommonMidPoint"];

            OPENCL_ASSERT(buildParameterArrayForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(buildParameterArrayForCommonMidPoint.setArg(argIndex++, sizeof(float), &minVelocity));
            OPENCL_ASSERT(buildParameterArrayForCommonMidPoint.setArg(argIndex++, sizeof(float), &increment));
            OPENCL_ASSERT(buildParameterArrayForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));

            commandQueue.enqueueNDRangeKernel(buildParameterArrayForCommonMidPoint, offset, global, local); 

            break;
        }
        case ZOCRS:
        case OCT:
        default:
           throw invalid_argument("Invalid traveltime model");
    }

    OPENCL_ASSERT(commandQueue.finish());
}

void OpenCLLinearSearchAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {
    cl_int errorCode;

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();
    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    int tauIndexDisplacement = gather->getTauIndexDisplacement();
    unsigned int windowSize = gather->getWindowSize();
    float apm = gather->getApm();

    vector<unsigned char> usedTraceMask(traceCount);

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    unique_ptr<cl::Buffer> deviceUsedTraceMaskArray = make_unique<cl::Buffer>(
        openClContext->getContext(),
        CL_MEM_READ_ONLY,
        traceCount * sizeof(unsigned char),
        &errorCode
    );

    OPENCL_ASSERT_CODE(errorCode);

    auto commandQueue = openClContext->getCommandQueue();

    OPENCL_ASSERT(commandQueue.enqueueFillBuffer<unsigned char>(
        *deviceUsedTraceMaskArray.get(),
        0,
        0,
        traceCount * sizeof(unsigned char)
    ));

    cl::NDRange offset;
    cl::NDRange global(static_cast<int>(ceil(static_cast<float>(traceCount) / static_cast<float>(threadCount))));
    cl::NDRange local(threadCount);

    LOGI("Using " << global.get()[0] << " blocks for traces filtering (threadCount = "<< threadCount << ")");

    chrono::duration<double> copyTime = chrono::duration<double>::zero();

    switch (traveltime->getModel()) {
        case CMP: {
            cl_uint argIndex = 0;
            cl::Kernel& selectTracesForCommonMidPoint = kernels["selectTracesForCommonMidPoint"];

            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::MDPNT])));
            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &traceCount));
            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, *deviceUsedTraceMaskArray.get()));
            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, sizeof(float), &m0));

            commandQueue.enqueueNDRangeKernel(selectTracesForCommonMidPoint, offset, global, local); 
            break;
        }
        case ZOCRS:
        case OCT: 
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

    LOGI("Execution time for copying traces is " << copyTime.count() << "s");
}
