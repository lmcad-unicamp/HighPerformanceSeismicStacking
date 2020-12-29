#include "common/include/output/Logger.hpp"
#include "common/include/semblance/result/ResultSet.hpp"

ResultSet::ResultSet(
    unsigned int numberOfResults,
    unsigned int samples
) : resultArray(numberOfResults), samplesPerResult(samples) {
}

const MidpointResult& ResultSet::getArrayForResult(unsigned int resultIndex) const {
    return resultArray[resultIndex];
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

const StatisticalMidpointResult& ResultSet::get(StatisticResult statResult) const {
    if (statisticalMidpointResult.find(statResult) != statisticalMidpointResult.end()) {
        return statisticalMidpointResult.at(statResult);
    }

    throw invalid_argument("Empty result for statistical result.");
};

void ResultSet::setStatisticalResultForMidpoint(float m0, StatisticResult stat, float statValue) {
    statisticalMidpointResult[stat].setStatisticalResultForMidpoint(m0, statValue);
}
