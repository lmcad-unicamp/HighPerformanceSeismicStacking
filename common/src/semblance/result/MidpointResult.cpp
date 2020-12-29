#include "common/include/semblance/result/MidpointResult.hpp"

using namespace std;

const map<float, vector<float>>& MidpointResult::getResultMap() const {
    return resultMap;
}

void MidpointResult::save(float m0, vector<float>::const_iterator start, vector<float>::const_iterator end) {
    resultMap[m0].assign(start, end);
}

const vector<float>& MidpointResult::get(float m0) const {
    if (resultMap.find(m0) != resultMap.end()) {
        return resultMap.at(m0);
    }

    throw invalid_argument("Empty result for given midpoint.");
}
