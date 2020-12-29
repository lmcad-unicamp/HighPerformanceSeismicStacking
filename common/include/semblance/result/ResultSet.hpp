#pragma once

#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/semblance/result/MidpointResult.hpp"
#include "common/include/semblance/result/StatisticalMidpointResult.hpp"

#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

using namespace std;

class ResultSet {
    private:
        vector<MidpointResult> resultArray;
        unordered_map<StatisticResult, StatisticalMidpointResult> statisticalMidpointResult;
        unsigned int samplesPerResult;

    public:
        ResultSet(unsigned int numberOfResults, unsigned int samples);

        const MidpointResult& getArrayForResult(unsigned int resultIndex) const;

        const StatisticalMidpointResult& get(StatisticResult statResult) const;

        void setAllResultsForMidpoint(float m0, const vector<float>& array);

        void setStatisticalResultForMidpoint(float m0, StatisticResult stat, float statValue);
};
