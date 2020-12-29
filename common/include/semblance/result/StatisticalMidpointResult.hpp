#pragma once

#include <map>

using namespace std;

class StatisticalMidpointResult {
    private:
        map<float, float> statisticalMidpointResult;
    public:
        float getStatisticalResultForMidpoint(float m0) const;
        float getAverageOfAllMidpoints() const;
        const map<float, float>& getStatisticalMidpointResultMap() const;

        void setStatisticalResultForMidpoint(float m0, float stat);

};
