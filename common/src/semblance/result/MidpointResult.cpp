#include "common/include/semblance/result/MidpointResult.hpp"

void MidpointResult::save(float m0, vector<float>::const_iterator start, vector<float>::const_iterator end) {
    MidpointResult[m0].assign(start, end);
}

const vector<float>& MidpointResult::get(float m0) const {
    if (MidpointResult.find(m0) != MidpointResult.end()) {
        return MidpointResult.at(m0);
    }

    throw invalid_argument("Empty result for given midpoint.");
}
