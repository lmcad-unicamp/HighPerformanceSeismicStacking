#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/traveltime/CommonMidPoint.hpp"
#include "common/include/traveltime/CommonReflectionSurface.hpp"
#include "common/include/traveltime/OffsetContinuationTrajectory.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLLinearSearchAlgorithm.hpp"
#include "opencl/include/semblance/data/OpenCLDataContainer.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <cmath>
#include <memory>
#include <random>

using namespace std;

OpenCLLinearSearchAlgorithm::OpenCLLinearSearchAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int threadCount
) : LinearSearchAlgorithm(traveltime, context, dataBuilder, threadCount), OpenCLComputeAlgorithm() {
}

void OpenCLLinearSearchAlgorithm::compileKernels(const string& deviceKernelSourcePath) {
    OpenCLComputeAlgorithm::compileKernels(deviceKernelSourcePath, "linear_search", traveltime, deviceContext);
}

void OpenCLLinearSearchAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {

    if (!filteredTracesCount) {
        LOGI("No trace has been selected for m0 = " << m0 << ". Skipping.");
        return;
    }

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    int tauIndexDisplacement = gather->getTauIndexDisplacement();
    unsigned int windowSize = gather->getWindowSize();
    float apm = gather->getApm();
    float h0 = traveltime->getReferenceHalfoffset();

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
        case OCT: {
            cl_uint argIndex = 0;
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
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceNotUsedCountArray)));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForOffsetContinuationTrajectory, offset, global, local));

            argIndex = 0;
            cl::Kernel& selectBestSemblancesForOffsetContinuationTrajectory = kernels["selectBestSemblancesForOffsetContinuationTrajectory"];

            OPENCL_ASSERT(selectBestSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL])));
            OPENCL_ASSERT(selectBestSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(commonResultDeviceArrayMap[SemblanceCommonResult::STACK])));
            OPENCL_ASSERT(selectBestSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(selectBestSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));
            OPENCL_ASSERT(selectBestSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(selectBestSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));

            cl::NDRange globalSelection(fitGlobal(samplesPerTrace, threadCount));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectBestSemblancesForOffsetContinuationTrajectory, offset, globalSelection, local));

            break;
        }
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
        case OCT: {
            float minVelocity = traveltime->getLowerBoundForParameter(OffsetContinuationTrajectory::VELOCITY);
            float incrementVelocity = discretizationStep[OffsetContinuationTrajectory::VELOCITY];
            unsigned int countVelocity = discretizationGranularity[OffsetContinuationTrajectory::VELOCITY];

            float minSlope = traveltime->getLowerBoundForParameter(OffsetContinuationTrajectory::SLOPE);
            float incrementSlope = discretizationStep[OffsetContinuationTrajectory::SLOPE];
            unsigned int countSlope = discretizationGranularity[OffsetContinuationTrajectory::SLOPE];

            cl_uint argIndex = 0;
            cl::Kernel& buildParameterArrayForOffsetContinuationTrajectory = kernels["buildParameterArrayForOffsetContinuationTrajectory"];

            OPENCL_ASSERT(buildParameterArrayForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(buildParameterArrayForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &minVelocity));
            OPENCL_ASSERT(buildParameterArrayForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &incrementVelocity));
            OPENCL_ASSERT(buildParameterArrayForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &countVelocity));
            OPENCL_ASSERT(buildParameterArrayForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &minSlope));
            OPENCL_ASSERT(buildParameterArrayForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &incrementSlope));
            OPENCL_ASSERT(buildParameterArrayForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &countSlope));
            OPENCL_ASSERT(buildParameterArrayForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &totalNumberOfParameters));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(buildParameterArrayForOffsetContinuationTrajectory, offset, global, local));

            break;
        }
        default:
           throw invalid_argument("Invalid traveltime model");
    }

    OPENCL_ASSERT(commandQueue.finish());
}

void OpenCLLinearSearchAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {
    cl_int errorCode;

    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
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
            unsigned int samplePop = 1024;
            vector<float> parameterSampleArray(samplePop * 2);
            float h0 = traveltime->getReferenceHalfoffset();

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

            cl_uint argIndex = 0;
            cl::Kernel& selectTracesForOffsetContinuationTrajectory = kernels["selectTracesForOffsetContinuationTrajectory"];

            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::MDPNT])));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::HLFOFFST])));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(selectionParameterArray)));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &samplePop));
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
