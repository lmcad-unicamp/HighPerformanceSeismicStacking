#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/traveltime/CommonMidPoint.hpp"
#include "common/include/traveltime/CommonReflectionSurface.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLLinearSearchAlgorithm.hpp"
#include "opencl/include/semblance/data/OpenCLDataContainer.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>

using namespace std;

OpenCLLinearSearchAlgorithm::OpenCLLinearSearchAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : LinearSearchAlgorithm(traveltime, context, dataBuilder) {
}

vector<string> OpenCLLinearSearchAlgorithm::readSourceFiles(const vector<string>& files) const {
    vector<string> result;
    for (auto file : files) {
        ifstream kernelFile(file);
        stringstream kernelSourceCode;
        kernelSourceCode << kernelFile.rdbuf();
        result.push_back(kernelSourceCode.str());
    }
    return result;
}

void OpenCLLinearSearchAlgorithm::compileKernels(const string& deviceKernelSourcePath) {
    cl_int errorCode;

    vector<string> kernelSources;
    vector<cl::Kernel> openClKernels;

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    const string traveltimeKernelPath = deviceKernelSourcePath + "/" + traveltime->getTraveltimeWord();

    filesystem::path specificKernelPath(traveltimeKernelPath + "/linear_search.cl");
    filesystem::path commonKernelPath(traveltimeKernelPath + "/common.cl");

    if (!filesystem::exists(specificKernelPath)) {
        throw runtime_error("Kernels for greedy algorithm do not exist.");
    }

    kernelSources.push_back(static_cast<string>(specificKernelPath));

    if (filesystem::exists(commonKernelPath)) {
        kernelSources.push_back(static_cast<string>(commonKernelPath));
    }

    cl::Program program(openClContext->getContext(), readSourceFiles(kernelSources), &errorCode);

    OPENCL_ASSERT_CODE(errorCode);

    errorCode = program.build({ openClContext->getDevice() }, "-cl-fast-relaxed-math -I../");

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

    LOGI("Computing semblance for m0 = " << m0);

    if (!filteredTracesCount) {
        LOGI("No trace has been selected for m0 = " << m0 << ". Skipping.");
        return;
    }

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();
    unsigned int numberOfCommonResults = traveltime->getNumberOfCommonResults();

    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    int tauIndexDisplacement = gather->getTauIndexDisplacement();
    unsigned int windowSize = gather->getWindowSize();

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(totalNumberOfParameters * samplesPerTrace, threadCount));
    cl::NDRange local(threadCount);

    switch (traveltime->getModel()) {
        case CMP: {
            cl_uint argIndex = 0;
            cl::Kernel& computeSemblancesForCommonMidPoint = kernels["computeSemblancesForCommonMidPoint"];

            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &filteredTracesCount));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(float), &dtInSeconds));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(int), &tauIndexDisplacement));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(int), &windowSize));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForCommonMidPoint, offset, global, local)); 

            argIndex = 0;
            cl::Kernel& selectBestSemblancesForCommonMidPoint = kernels["selectBestSemblancesForCommonMidPoint"];

            OPENCL_ASSERT(selectBestSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(selectBestSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));
            OPENCL_ASSERT(selectBestSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(selectBestSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(selectBestSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(selectBestSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));

            cl::NDRange globalSelection(fitGlobal(samplesPerTrace, threadCount));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectBestSemblancesForCommonMidPoint, offset, globalSelection, local)); 

            break;
        }
        case ZOCRS: {
            cl_uint argIndex = 0;
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
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForZeroOffsetCommonReflectionSurface, offset, global, local)); 

            argIndex = 0;
            cl::Kernel& selectBestSemblancesForZeroOffsetCommonReflectionSurface = kernels["selectBestSemblancesForZeroOffsetCommonReflectionSurface"];

            OPENCL_ASSERT(selectBestSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(selectBestSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));
            OPENCL_ASSERT(selectBestSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(selectBestSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(selectBestSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(selectBestSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));

            cl::NDRange globalSelection(fitGlobal(samplesPerTrace, threadCount));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectBestSemblancesForZeroOffsetCommonReflectionSurface, offset, globalSelection, local)); 

            break;
        }
        case OCT:
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    OPENCL_ASSERT(commandQueue.finish());
}

void OpenCLLinearSearchAlgorithm::initializeParameters() {

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(totalNumberOfParameters, threadCount));
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

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(buildParameterArrayForCommonMidPoint, offset, global, local)); 

            break;
        }
        case ZOCRS: {
            float minVelocity = traveltime->getLowerBoundForParameter(CommonReflectionSurface::VELOCITY);
            float incrementVelocity = discretizationStep[CommonReflectionSurface::VELOCITY];
            unsigned int countVelocity = discretizationGranularity[CommonReflectionSurface::VELOCITY];

            float minA = traveltime->getLowerBoundForParameter(CommonReflectionSurface::A);
            float incrementA = discretizationStep[CommonReflectionSurface::A];
            unsigned int countA = discretizationGranularity[CommonReflectionSurface::A];

            float minB = traveltime->getLowerBoundForParameter(CommonReflectionSurface::B);
            float incrementB = discretizationStep[CommonReflectionSurface::B];
            unsigned int countB = discretizationGranularity[CommonReflectionSurface::B];

            cl_uint argIndex = 0;
            cl::Kernel& buildParameterArrayForZeroOffsetCommonReflectionSurface = kernels["buildParameterArrayForZeroOffsetCommonReflectionSurface"];

            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &minVelocity));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &incrementVelocity));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &countVelocity));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &minA));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &incrementA));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &countA));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &minB));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &incrementB));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &countB));
            OPENCL_ASSERT(buildParameterArrayForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(buildParameterArrayForZeroOffsetCommonReflectionSurface, offset, global, local)); 

            break;
        }
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
    auto commandQueue = openClContext->getCommandQueue();

    unique_ptr<cl::Buffer> deviceUsedTraceMaskArray = make_unique<cl::Buffer>(
        openClContext->getContext(),
        CL_MEM_READ_ONLY,
        traceCount * sizeof(unsigned char),
        nullptr,
        &errorCode
    );

    OPENCL_ASSERT_CODE(errorCode);

    OPENCL_ASSERT(commandQueue.enqueueFillBuffer(
        *deviceUsedTraceMaskArray.get(),
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
            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, *deviceUsedTraceMaskArray.get()));
            OPENCL_ASSERT(selectTracesForCommonMidPoint.setArg(argIndex++, sizeof(float), &m0));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectTracesForCommonMidPoint, offset, global, local));
            break;
        }
        case ZOCRS: {
            cl_uint argIndex = 0;
            cl::Kernel& selectTracesForZeroOffsetCommonReflectionSurface = kernels["selectTracesForZeroOffsetCommonReflectionSurface"];

            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::MDPNT])));
            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &traceCount));
            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, *deviceUsedTraceMaskArray.get()));
            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &m0));
            OPENCL_ASSERT(selectTracesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &apm));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectTracesForZeroOffsetCommonReflectionSurface, offset, global, local));
            break;
        }
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
