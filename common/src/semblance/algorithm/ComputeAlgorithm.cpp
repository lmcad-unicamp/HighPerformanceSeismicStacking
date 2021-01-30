#include "common/include/execution/Utils.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/output/Logger.hpp"

#include <numeric>

using namespace std;

ComputeAlgorithm::ComputeAlgorithm(
    const string& name,
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int threadCount
) : isSet(false),
    dataFactory(dataBuilder),
    algorithmName(name),
    deviceContext(context),
    traveltime(model),
    threadCount(threadCount),
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
    unsigned int samplesPerTrace = gather->getSamplesPerTrace();

    deviceFilteredTracesDataMap[GatherData::SAMPL].reset(dataFactory->build(traceCount * samplesPerTrace, deviceContext));

    deviceFilteredTracesDataMap[GatherData::MDPNT].reset(dataFactory->build(traceCount, deviceContext));

    deviceFilteredTracesDataMap[GatherData::HLFOFFST].reset(dataFactory->build(traceCount, deviceContext));

    deviceFilteredTracesDataMap[GatherData::HLFOFFST_SQ].reset(dataFactory->build(traceCount, deviceContext));

    vector<float> tempMidpointArray(traceCount), tempHalfoffsetArray(traceCount), tempHalfoffsetSqArray(traceCount);

    for (unsigned int t = 0; t < traceCount; t++) {
        const Trace& trace = gather->getTraceAtIndex(t);

        deviceFilteredTracesDataMap[GatherData::SAMPL]->copyFromWithOffset(trace.getSamples(), t * samplesPerTrace);

        float m = gather->getMidpointOfTrace(t);
        float h = gather->getHalfoffsetOfTrace(t);

        tempMidpointArray[t] = m;
        tempHalfoffsetArray[t] = h;
        tempHalfoffsetSqArray[t] = h * h;
    }

    deviceFilteredTracesDataMap[GatherData::MDPNT]->copyFrom(tempMidpointArray);
    deviceFilteredTracesDataMap[GatherData::HLFOFFST]->copyFrom(tempHalfoffsetArray);
    deviceFilteredTracesDataMap[GatherData::HLFOFFST_SQ]->copyFrom(tempHalfoffsetSqArray);

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
    float totalUsedTracesCount,
    chrono::duration<double> totalExecutionTime,
    chrono::duration<double> selectionExecutionTime
) {
    Gather* gather = Gather::getInstance();

    chrono::duration<double> statisticalDuration = chrono::duration<double>::zero();

    chrono::steady_clock::time_point stasticalStartTime = chrono::steady_clock::now();

    vector<float> tempNotUsedCount(deviceNotUsedCountArray->getElementCount());

    deviceNotUsedCountArray->pasteTo(tempNotUsedCount);

    float notUsedTracesCount, interpolationsPerformed;

    notUsedTracesCount = accumulate(tempNotUsedCount.begin(), tempNotUsedCount.end(), 0.0f);

    interpolationsPerformed = static_cast<float>(gather->getWindowSize()) * (totalUsedTracesCount - notUsedTracesCount);

    computedStatisticalResults[StatisticResult::INTR_PER_SEC] = interpolationsPerformed / static_cast<float>(totalExecutionTime.count());

    computedStatisticalResults[StatisticResult::EFFICIENCY] = 1.0f;

    if (totalUsedTracesCount > 0) {
        computedStatisticalResults[StatisticResult::EFFICIENCY] -= notUsedTracesCount / totalUsedTracesCount;
    }

    computedStatisticalResults[StatisticResult::SELECTED_TRACES] = static_cast<float>(filteredTracesCount);

    computedStatisticalResults[StatisticResult::TOTAL_SELECTION_KERNEL_EXECUTION_TIME] = selectionExecutionTime.count();

    computedStatisticalResults[StatisticResult::TOTAL_KERNEL_EXECUTION_TIME] = totalExecutionTime.count();

    statisticalDuration += chrono::steady_clock::now() - stasticalStartTime;

    computedStatisticalResults[StatisticResult::TOTAL_STATISTIC_COMPUTE_TIME] = statisticalDuration.count();

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

    LOGH("Selected " << filteredTracesCount << " traces");

    /* Reallocate filtered sample array */
    deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]->reallocate(filteredTracesCount * gather->getSamplesPerTrace());
    deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]->reallocate(filteredTracesCount);
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]->reallocate(filteredTracesCount);
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST_SQ]->reallocate(filteredTracesCount);

    vector<float> tempMidpoint(filteredTracesCount);
    vector<float> tempHalfoffset(filteredTracesCount);
    vector<float> tempHalfoffsetSquared(filteredTracesCount);

    unsigned int cudaArrayOffset = 0, idx = 0;
    for (unsigned int i = 0; i < traceCount && filteredTracesCount; i++) {

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

    computedStatisticalResults[StatisticResult::TOTAL_COPY_TIME] = copyExecutionTime.count();
}

void ComputeAlgorithm::compileKernels(const string& deviceKernelSourcePath) {
}

pair<unsigned int, unsigned int> ComputeAlgorithm::selectTracesContinuous(float m0) const {

    pair<unsigned int, unsigned int> traceRange(0, 0);

    Gather* gather = Gather::getInstance();
    float apm = gather->getApm();

    const map<float, Cdp>& cdps = gather->getCdps();
    unsigned int traceIndex = 0;
    bool foundStartIndex = false;

    for (auto it = cdps.begin(); it != cdps.end(); it++) {
        float m = it->first;
        const Cdp& cdp = it->second;
        bool useTrace = false;

        switch (traveltime->getModel()) {
            case CMP:
                useTrace = (m0 == m);
                break;
            case ZOCRS:
                useTrace = fabs(m0 - m) <= apm;
                break;
            default:
                useTrace = false;
        }

        if (useTrace) {
            if (!foundStartIndex) {
                traceRange.first = traceIndex;
                foundStartIndex = true;
            }
            traceRange.second += cdp.getTraceCount();
        }
        traceIndex += cdp.getTraceCount();
    }

    return traceRange;
}
