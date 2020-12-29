#include "common/include/output/Logger.hpp"
#include "common/include/semblance/result/ResultSet.hpp"

using namespace std;

ResultSet::ResultSet(
    unsigned int numberOfResults,
    unsigned int samples
) : resultArray(numberOfResults), samplesPerResult(samples) {
}

void ResultSet::copyFrom(const ResultSet& other) {
    for(unsigned int resultIndex = 0; resultIndex < resultArray.size(); resultIndex++) {
        const MidpointResult& midpointResult = other.get(resultIndex);
        const map<float, vector<float>>& resultMap = midpointResult.getResultMap();

        for (auto it = resultMap.cbegin(); it != resultMap.cend(); it++) {
            float m0 = it->first;
            const vector<float>& resultArray = it->second;
            setResultForMidpoint(m0, resultIndex, resultArray);
        }
    }

    auto statisticResultMap = other.getStatisticalMidpointResultMap();
    for (auto it = statisticResultMap.cbegin(); it != statisticResultMap.cend(); it++) {
        const StatisticalMidpointResult& statisticalMidpointResult = it->second;
        const map<float, float>& statisticalMidpointResultMap = statisticalMidpointResult.getStatisticalMidpointResultMap();
        StatisticResult stat = it->first;

        for (auto midpointIt = statisticalMidpointResultMap.cbegin(); midpointIt != statisticalMidpointResultMap.cend(); midpointIt++) {
            float m0 = midpointIt->first;
            float statValue = midpointIt->second;
            setStatisticalResultForMidpoint(m0, stat, statValue);
        }
    }
}

const MidpointResult& ResultSet::get(unsigned int resultIndex) const {
    return resultArray[resultIndex];
}

const StatisticalMidpointResult& ResultSet::get(StatisticResult statResult) const {
    if (statisticalMidpointResult.find(statResult) != statisticalMidpointResult.end()) {
        return statisticalMidpointResult.at(statResult);
    }

    throw invalid_argument("Empty result for statistical result.");
}

const unordered_map<StatisticResult, StatisticalMidpointResult>& ResultSet::getStatisticalMidpointResultMap() const {
    return statisticalMidpointResult;
}

void ResultSet::setResultForMidpoint(float m0, unsigned int resultIndex, const vector<float>& array) {
    if (resultIndex >= resultArray.size()) {
        throw invalid_argument("Result index is out of range.");
    }

    resultArray[resultIndex].save(m0, array.cbegin(), array.cend());
}

void ResultSet::setAllResultsForMidpoint(float m0, const vector<float>& array) {

    vector<float>::const_iterator resultIterator = array.cbegin();

    for(unsigned int i = 0; i < resultArray.size(); i++) {

        auto start = resultIterator + i * samplesPerResult;
        auto end = start + samplesPerResult;

        LOGD("Writing " << samplesPerResult << " elements for i = " << i);

        resultArray[i].save(m0, start, end);
    }
}

void ResultSet::setStatisticalResultForMidpoint(float m0, StatisticResult stat, float statValue) {
    statisticalMidpointResult[stat].setStatisticalResultForMidpoint(m0, statValue);
}
