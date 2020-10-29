#pragma once

#include "common/include/model/Gather.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/semblance/result/MidpointResult.hpp"
#include "common/include/semblance/result/StatisticalMidpointResult.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <map>
#include <memory>
#include <string>

using namespace std;

class Dumper {
    private:
        string outputDirectoryPath;

    public:
        Dumper(const string& path, const string& dataFile, const string& computeMethod, const string& traveltime);

        void createDir() const;

        void dumpAlgorithm(ComputeAlgorithm* algorithm) const;

        void dumpGatherParameters(const string& file) const;

        void dumpResult(const string& resultName, const MidpointResult& result) const;

        void dumpTraveltime(Traveltime* model) const;

        void dumpStatisticalResult(const string& statResultName, const StatisticalMidpointResult& statResult) const;

        const string& getOutputDirectoryPath() const;
};
