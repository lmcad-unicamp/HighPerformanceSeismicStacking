#pragma once

#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"
#include "common/include/semblance/algorithm/DifferentialEvolutionAlgorithm.hpp"
#include "common/include/semblance/algorithm/StretchFreeAlgorithm.hpp"

#include <memory>

using namespace std;

class ComputeAlgorithmBuilder {
    public:
        virtual LinearSearchAlgorithm* buildLinearSearchAlgorithm(
                shared_ptr<Traveltime> traveltime,
                shared_ptr<DeviceContext> context,
                unsigned int threadCount,
                const vector<int>& discretizationArray
        ) = 0;

        virtual DifferentialEvolutionAlgorithm* buildDifferentialEvolutionAlgorithm(
                shared_ptr<Traveltime> traveltime,
                shared_ptr<DeviceContext> context,
                unsigned int threadCount,
                unsigned int generation,
                unsigned int individualsPerPopulation
        ) = 0;

        virtual StretchFreeAlgorithm* buildStretchFreeAlgorithm(
                shared_ptr<Traveltime> traveltime,
                shared_ptr<DeviceContext> context,
                unsigned int threadCount,
                const vector<string>& parameterFileArray
        ) = 0;
};
