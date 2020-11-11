#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLDifferentialEvolutionAlgorithm.hpp"
#include "opencl/include/semblance/data/OpenCLDataContainer.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <stdlib.h>
#include <time.h>

OpenCLDifferentialEvolutionAlgorithm::OpenCLDifferentialEvolutionAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int gen,
    unsigned int ind
) : DifferentialEvolutionAlgorithm(model, context, dataBuilder, gen, ind), OpenCLComputeAlgorithm() {
}

void OpenCLDifferentialEvolutionAlgorithm::compileKernels(const string& deviceKernelSourcePath) {
    OpenCLComputeAlgorithm::compileKernels(deviceKernelSourcePath, "differential_evolution", traveltime, deviceContext);
}

void OpenCLDifferentialEvolutionAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {
    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    int tauIndexDisplacement = gather->getTauIndexDisplacement();
    unsigned int windowSize = gather->getWindowSize();
    float apm = gather->getApm();
    float h0 = traveltime->getReferenceHalfoffset();
    unsigned int numberOfCommonResults = traveltime->getNumberOfCommonResults();

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(samplesPerTrace * individualsPerPopulation, threadCount));
    cl::NDRange local(threadCount);

    switch (traveltime->getModel()) {
        case CMP: {
            cl_uint argIndex = 0;
            cl::Kernel& computeSemblancesForCommonMidPoint = kernels["computeSemblancesForCommonMidPoint"];

            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_SAMPL])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ])));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &filteredTracesCount));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(float), &dtInSeconds));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(int), &tauIndexDisplacement));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(int), &windowSize));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &numberOfCommonResults));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForCommonMidPoint, offset, global, local));

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
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &m0));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(float), &dtInSeconds));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(int), &tauIndexDisplacement));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(int), &windowSize));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &numberOfCommonResults));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForZeroOffsetCommonReflectionSurface, offset, global, local));

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
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &apm));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &m0));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &h0));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &dtInSeconds));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(int), &tauIndexDisplacement));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(int), &windowSize));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &numberOfCommonResults));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceNotUsedCountArray)));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(computeSemblancesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(computeSemblancesForOffsetContinuationTrajectory, offset, global, local));

            break;
        }
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    OPENCL_ASSERT(commandQueue.finish());
}

void OpenCLDifferentialEvolutionAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {
    LOGI("Selecting traces for m0 = " << m0);

    cl_int errorCode;

    Gather* gather = Gather::getInstance();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    unsigned int traceCount = gather->getTotalTracesCount();
    float dtInSeconds = gather->getSamplePeriodInSeconds();
    float apm = gather->getApm();
    float h0 = traveltime->getReferenceHalfoffset();

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
            cl_uint argIndex = 0;
            cl::Kernel& selectTracesForOffsetContinuationTrajectory = kernels["selectTracesForOffsetContinuationTrajectory"];

            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::MDPNT])));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceFilteredTracesDataMap[GatherData::HLFOFFST])));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceParameterArray)));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &traceCount));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
            OPENCL_ASSERT(selectTracesForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(float), &dtInSeconds));
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
        *deviceUsedTraceMaskArray,
        CL_TRUE,
        0,
        usedTraceMask.size() * sizeof(unsigned char),
        usedTraceMask.data()
    ));

    MEASURE_EXEC_TIME(copyTime, copyOnlySelectedTracesToDevice(usedTraceMask));

    LOGI("Execution time for copying traces is " << copyTime.count() << "s");
}

void OpenCLDifferentialEvolutionAlgorithm::setupRandomSeedArray() {
    cl_int errorCode;

    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    unsigned int seedCount = samplesPerTrace * individualsPerPopulation;

    st.reset(new cl::Buffer(
        openClContext->getContext(),
        CL_MEM_READ_WRITE,
        seedCount * sizeof(unsigned int),
        nullptr,
        &errorCode
    ));

    OPENCL_ASSERT_CODE(errorCode);

    vector<unsigned int> seedArray(seedCount);

    srand(static_cast<unsigned int>(time(NULL)));

    for (unsigned int i = 0; i < seedCount; i++) {
        seedArray[i] = static_cast<unsigned int>(rand());
    }

    OPENCL_ASSERT(commandQueue.enqueueWriteBuffer(
        *st,
        CL_TRUE,
        0,
        seedCount * sizeof(unsigned int),
        seedArray.data()
    ));
}

void OpenCLDifferentialEvolutionAlgorithm::startAllPopulations() {
    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(samplesPerTrace * individualsPerPopulation, threadCount));
    cl::NDRange local(threadCount);

    cl_uint argIndex = 0;
    cl::Kernel startPopulations = kernels["startPopulations"];

    OPENCL_ASSERT(startPopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(x)));
    OPENCL_ASSERT(startPopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(min)));
    OPENCL_ASSERT(startPopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(max)));
    OPENCL_ASSERT(startPopulations.setArg(argIndex++, *st));
    OPENCL_ASSERT(startPopulations.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
    OPENCL_ASSERT(startPopulations.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
    OPENCL_ASSERT(startPopulations.setArg(argIndex++, sizeof(unsigned int), &numberOfParameters));

    OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(startPopulations, offset, global, local));

    OPENCL_ASSERT(commandQueue.finish());

    fx->reset();
    fu->reset();
}

void OpenCLDifferentialEvolutionAlgorithm::mutateAllPopulations() {

    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(samplesPerTrace * individualsPerPopulation, threadCount));
    cl::NDRange local(threadCount);

    cl_uint argIndex = 0;
    cl::Kernel mutatePopulations = kernels["mutatePopulations"];

    OPENCL_ASSERT(mutatePopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(v)));
    OPENCL_ASSERT(mutatePopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(x)));
    OPENCL_ASSERT(mutatePopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(min)));
    OPENCL_ASSERT(mutatePopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(max)));
    OPENCL_ASSERT(mutatePopulations.setArg(argIndex++, *st));
    OPENCL_ASSERT(mutatePopulations.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
    OPENCL_ASSERT(mutatePopulations.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
    OPENCL_ASSERT(mutatePopulations.setArg(argIndex++, sizeof(unsigned int), &numberOfParameters));

    OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(mutatePopulations, offset, global, local));

    OPENCL_ASSERT(commandQueue.finish());
}

void OpenCLDifferentialEvolutionAlgorithm::crossoverPopulationIndividuals() {
    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    unsigned int numberOfParameters = traveltime->getNumberOfParameters();

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(samplesPerTrace * individualsPerPopulation, threadCount));
    cl::NDRange local(threadCount);

    cl_uint argIndex = 0;
    cl::Kernel crossoverPopulations = kernels["crossoverPopulations"];

    OPENCL_ASSERT(crossoverPopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(u)));
    OPENCL_ASSERT(crossoverPopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(x)));
    OPENCL_ASSERT(crossoverPopulations.setArg(argIndex++, OPENCL_DEV_BUFFER(v)));
    OPENCL_ASSERT(crossoverPopulations.setArg(argIndex++, *st));
    OPENCL_ASSERT(crossoverPopulations.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
    OPENCL_ASSERT(crossoverPopulations.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
    OPENCL_ASSERT(crossoverPopulations.setArg(argIndex++, sizeof(unsigned int), &numberOfParameters));

    OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(crossoverPopulations, offset, global, local));
    OPENCL_ASSERT(commandQueue.finish());
}

void OpenCLDifferentialEvolutionAlgorithm::advanceGeneration() {
    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfCommonResults = traveltime->getNumberOfCommonResults();

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(samplesPerTrace * individualsPerPopulation, threadCount));
    cl::NDRange local(threadCount);

    cl_uint argIndex = 0;
    cl::Kernel nextGeneration = kernels["nextGeneration"];

    OPENCL_ASSERT(nextGeneration.setArg(argIndex++, OPENCL_DEV_BUFFER(x)));
    OPENCL_ASSERT(nextGeneration.setArg(argIndex++, OPENCL_DEV_BUFFER(fx)));
    OPENCL_ASSERT(nextGeneration.setArg(argIndex++, OPENCL_DEV_BUFFER(u)));
    OPENCL_ASSERT(nextGeneration.setArg(argIndex++, OPENCL_DEV_BUFFER(fu)));
    OPENCL_ASSERT(nextGeneration.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
    OPENCL_ASSERT(nextGeneration.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
    OPENCL_ASSERT(nextGeneration.setArg(argIndex++, sizeof(unsigned int), &numberOfParameters));
    OPENCL_ASSERT(nextGeneration.setArg(argIndex++, sizeof(unsigned int), &numberOfCommonResults));

    OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(nextGeneration, offset, global, local));
    OPENCL_ASSERT(commandQueue.finish());
}

void OpenCLDifferentialEvolutionAlgorithm::selectBestIndividuals(vector<float>& resultArrays) {
    Gather* gather = Gather::getInstance();

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);
    auto commandQueue = openClContext->getCommandQueue();

    unsigned int samplesPerTrace = gather->getSamplesPerTrace();
    unsigned int numberOfCommonResults = traveltime->getNumberOfCommonResults();

    cl::NDRange offset;
    cl::NDRange global(fitGlobal(samplesPerTrace, threadCount));
    cl::NDRange local(threadCount);

    switch (traveltime->getModel()) {
        case CMP: {
            cl_uint argIndex = 0;
            cl::Kernel& selectBestIndividualsForCommonMidPoint = kernels["selectBestIndividualsForCommonMidPoint"];

            OPENCL_ASSERT(selectBestIndividualsForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(x)));
            OPENCL_ASSERT(selectBestIndividualsForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(fx)));
            OPENCL_ASSERT(selectBestIndividualsForCommonMidPoint.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));
            OPENCL_ASSERT(selectBestIndividualsForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
            OPENCL_ASSERT(selectBestIndividualsForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(selectBestIndividualsForCommonMidPoint.setArg(argIndex++, sizeof(unsigned int), &numberOfCommonResults));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectBestIndividualsForCommonMidPoint, offset, global, local));
            break;
        }
        case ZOCRS: {
            cl_uint argIndex = 0;
            cl::Kernel& selectBestIndividualsForZeroOffsetCommonReflectionSurface = kernels["selectBestIndividualsForZeroOffsetCommonReflectionSurface"];

            OPENCL_ASSERT(selectBestIndividualsForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(x)));
            OPENCL_ASSERT(selectBestIndividualsForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(fx)));
            OPENCL_ASSERT(selectBestIndividualsForZeroOffsetCommonReflectionSurface.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));
            OPENCL_ASSERT(selectBestIndividualsForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
            OPENCL_ASSERT(selectBestIndividualsForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(selectBestIndividualsForZeroOffsetCommonReflectionSurface.setArg(argIndex++, sizeof(unsigned int), &numberOfCommonResults));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectBestIndividualsForZeroOffsetCommonReflectionSurface, offset, global, local));
            break;
        }
        case OCT: {
            cl_uint argIndex = 0;
            cl::Kernel& selectBestIndividualsForOffsetContinuationTrajectory = kernels["selectBestIndividualsForOffsetContinuationTrajectory"];

            OPENCL_ASSERT(selectBestIndividualsForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(x)));
            OPENCL_ASSERT(selectBestIndividualsForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(fx)));
            OPENCL_ASSERT(selectBestIndividualsForOffsetContinuationTrajectory.setArg(argIndex++, OPENCL_DEV_BUFFER(deviceResultArray)));
            OPENCL_ASSERT(selectBestIndividualsForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &individualsPerPopulation));
            OPENCL_ASSERT(selectBestIndividualsForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &samplesPerTrace));
            OPENCL_ASSERT(selectBestIndividualsForOffsetContinuationTrajectory.setArg(argIndex++, sizeof(unsigned int), &numberOfCommonResults));

            OPENCL_ASSERT(commandQueue.enqueueNDRangeKernel(selectBestIndividualsForOffsetContinuationTrajectory, offset, global, local));
            break;
        }
        default:
            throw invalid_argument("Invalid traveltime model");
    }

    OPENCL_ASSERT(commandQueue.finish());

    deviceResultArray->pasteTo(resultArrays);
}
