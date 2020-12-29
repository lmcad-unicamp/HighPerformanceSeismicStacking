#include "common/include/semblance/result/StatisticalMidpointResult.hpp"

using namespace std;

float StatisticalMidpointResult::getStatisticalResultForMidpoint(float m0) const {
    if (statisticalMidpointResult.find(m0) != statisticalMidpointResult.end()) {
        return statisticalMidpointResult.at(m0);
    }

    throw invalid_argument("Empty result for given midpoint.");
}

void StatisticalMidpointResult::setStatisticalResultForMidpoint(float m0, float stat) {
    statisticalMidpointResult[m0] = stat;
}

float StatisticalMidpointResult::getAverageOfAllMidpoints() const {

    if (statisticalMidpointResult.empty()) {
        return 0;
    }

    float sum = 0;
    float count = static_cast<float>(statisticalMidpointResult.size());

    for (auto iterator = statisticalMidpointResult.begin(); iterator != statisticalMidpointResult.end(); iterator++) {
        float statisticalResult = iterator->second;
        if (statisticalResult != 0) {
            sum += statisticalResult;
        }
    }

    return sum / count;
}

const map<float, float>& StatisticalMidpointResult::getStatisticalMidpointResultMap() const {
    return statisticalMidpointResult;
}
