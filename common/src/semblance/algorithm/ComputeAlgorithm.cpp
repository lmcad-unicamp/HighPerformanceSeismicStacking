#include "common/include/execution/Utils.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/output/Logger.hpp"

#include <numeric>

using namespace std;

ComputeAlgorithm::ComputeAlgorithm(
    const string& name,
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : isSet(false),
    dataFactory(dataBuilder),
    algorithmName(name),
    deviceContext(context),
    traveltime(model),
    threadCount(64),
    threadCountToRestore(threadCount),
    computedStatisticalResults(static_cast<unsigned int>(StatisticResult::CNT)) {
}

ComputeAlgorithm::~ComputeAlgorithm() {
}

bool ComputeAlgorithm::isSetUp() const {
    return isSet;
}

void ComputeAlgorithm::copyGatherDataToDevice() {
    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    LOGH("Allocating data for GatherData::MDPNT [" << traceCount << " elements]");

    deviceFilteredTracesDataMap[GatherData::MDPNT].reset(dataFactory->build(traceCount, deviceContext));

    LOGH("Allocating data for GatherData::HLFOFFST [" << traceCount << " elements]");

    deviceFilteredTracesDataMap[GatherData::HLFOFFST].reset(dataFactory->build(traceCount, deviceContext));

    vector<float> tempMidpointArray(traceCount), tempHalfoffsetArray(traceCount);

    for (unsigned int t = 0; t < traceCount; t++) {
        tempMidpointArray[t] = gather->getMidpointOfTrace(t);
        tempHalfoffsetArray[t] = gather->getHalfoffsetOfTrace(t);
    }

    deviceFilteredTracesDataMap[GatherData::MDPNT]->copyFrom(tempMidpointArray);
    deviceFilteredTracesDataMap[GatherData::HLFOFFST]->copyFrom(tempHalfoffsetArray);

    deviceFilteredTracesDataMap[GatherData::FILT_SAMPL].reset(dataFactory->build(deviceContext));
    deviceFilteredTracesDataMap[GatherData::FILT_MDPNT].reset(dataFactory->build(deviceContext));
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST].reset(dataFactory->build(deviceContext));
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ].reset(dataFactory->build(deviceContext));
}

float ComputeAlgorithm::getStatisticalResult(StatisticResult statResult) const {
    if (computedStatisticalResults.find(statResult) == computedStatisticalResults.end()) {
        throw invalid_argument("Couldn't find value for statistical result.");
    }

    return computedStatisticalResults.at(statResult);
}

void ComputeAlgorithm::saveStatisticalResults(
    unsigned long totalUsedTracesCount,
    chrono::duration<double> totalExecutionTime,
    chrono::duration<double> selectionExecutionTime
) {
    Gather* gather = Gather::getInstance();

    vector<float> tempNotUsedCount(deviceNotUsedCountArray->getElementCount());

    deviceNotUsedCountArray->pasteTo(tempNotUsedCount);

    unsigned long notUsedTracesCount, interpolationsPerformed;

    notUsedTracesCount = static_cast<unsigned long>(
        accumulate(tempNotUsedCount.begin(), tempNotUsedCount.end(), 0.0f)
    );

    interpolationsPerformed = static_cast<unsigned long>(gather->getWindowSize()) *
        (totalUsedTracesCount - notUsedTracesCount);

    computedStatisticalResults[StatisticResult::INTR_PER_SEC] =
        static_cast<float>(interpolationsPerformed) / static_cast<float>(totalExecutionTime.count());

    computedStatisticalResults[StatisticResult::EFFICIENCY] = 1.0f;

    if (totalUsedTracesCount > 0) {
        computedStatisticalResults[StatisticResult::EFFICIENCY] -=
            static_cast<float>(notUsedTracesCount) / static_cast<float>(totalUsedTracesCount);
    }

    computedStatisticalResults[StatisticResult::SELECTED_TRACES] = static_cast<float>(filteredTracesCount);

    computedStatisticalResults[StatisticResult::TOTAL_SELECTION_KERNEL_EXECUTION_TIME] = selectionExecutionTime.count();

    computedStatisticalResults[StatisticResult::TOTAL_KERNEL_EXECUTION_TIME] = totalExecutionTime.count();

    LOGI("Total execution time for selecting traces is " << selectionExecutionTime.count() << "s");

    LOGI("Total execution time for kernels is " << totalExecutionTime.count() << "s");
}

void ComputeAlgorithm::copyOnlySelectedTracesToDevice(
    const vector<unsigned char>& usedTraceMask
) {
    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    chrono::duration<double> copyExecutionTime = chrono::duration<double>::zero();

    filteredTracesCount = accumulate(usedTraceMask.begin(), usedTraceMask.end(), 0);

    LOGI("Selected " << filteredTracesCount << " traces");

    /* Reallocate filtered sample array */
    deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]->reallocate(filteredTracesCount * gather->getSamplesPerTrace());
    deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]->reallocate(filteredTracesCount);
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]->reallocate(filteredTracesCount);
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ]->reallocate(filteredTracesCount);

    vector<float> tempMidpoint(filteredTracesCount);
    vector<float> tempHalfoffset(filteredTracesCount);
    vector<float> tempHalfoffsetSquared(filteredTracesCount);

    unsigned int cudaArrayOffset = 0, idx = 0;
    for (unsigned int i = 0; i < traceCount; i++) {
        if (usedTraceMask[i]) {

            const Trace& trace = gather->getTraceAtIndex(i);

            MEASURE_EXEC_TIME(copyExecutionTime, deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]->copyFromWithOffset(trace.getSamples(), cudaArrayOffset));

            tempMidpoint[idx] = trace.getMidpoint();
            tempHalfoffset[idx] = trace.getHalfoffset();
            tempHalfoffsetSquared[idx] = trace.getHalfoffset() * trace.getHalfoffset();

            cudaArrayOffset += gather->getSamplesPerTrace();
            idx++;
        }
    }

    deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]->copyFrom(tempMidpoint);
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]->copyFrom(tempHalfoffset);
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ]->copyFrom(tempHalfoffsetSquared);

    LOGI("Copy time is " << copyExecutionTime.count());
}

void ComputeAlgorithm::changeThreadCountTemporarilyTo(unsigned int t) {
    LOGD("Updating threadCount to " << t);
    threadCount = t;
}

void ComputeAlgorithm::restoreThreadCount() {
    LOGD("Restoring threadCount to " << threadCountToRestore);
    threadCount = threadCountToRestore;
}

void ComputeAlgorithm::setDeviceSourcePath(const string& path) {
    deviceSource = path;
}
