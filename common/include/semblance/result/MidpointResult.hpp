#pragma once

#include <map>
#include <vector>

using namespace std;

class MidpointResult {
    private:
        map<float, vector<float>> MidpointResult;
    public:
        const vector<float>& get(float m0) const;
        void save(float m0, vector<float>::const_iterator start, vector<float>::const_iterator end);
};
