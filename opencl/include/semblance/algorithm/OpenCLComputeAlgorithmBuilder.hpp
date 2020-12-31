#pragma once

#include "common/include/semblance/algorithm/ComputeAlgorithmBuilder.hpp"

#include <memory>

using namespace std;

class OpenCLComputeAlgorithmBuilder : public ComputeAlgorithmBuilder {
    protected:
        static unique_ptr<ComputeAlgorithmBuilder> instance;
    public:
        static ComputeAlgorithmBuilder* getInstance();

        LinearSearchAlgorithm* buildLinearSearchAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            unsigned int threadCount,
            const vector<int>& discretizationArray
        ) override;

        DifferentialEvolutionAlgorithm* buildDifferentialEvolutionAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            unsigned int threadCount,
            unsigned int generation,
            unsigned int individualsPerPopulation
        ) override;

        StretchFreeAlgorithm* buildStretchFreeAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            unsigned int threadCount,
            const vector<string>& parameterFileArray
        ) override;
};
