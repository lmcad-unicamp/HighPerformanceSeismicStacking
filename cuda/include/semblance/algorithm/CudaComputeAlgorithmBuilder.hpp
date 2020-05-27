#pragma once

#include "common/include/semblance/algorithm/ComputeAlgorithmBuilder.hpp"
#include "common/include/semblance/data/DeviceContext.hpp"

#include <memory>

using namespace std;

class CudaComputeAlgorithmBuilder : public ComputeAlgorithmBuilder {
    protected:
        static unique_ptr<ComputeAlgorithmBuilder> instance;
    public:
        static ComputeAlgorithmBuilder* getInstance();

        LinearSearchAlgorithm* buildLinearSearchAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            const vector<int>& discretizationArray
        ) override;

        DifferentialEvolutionAlgorithm* buildDifferentialEvolutionAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            unsigned int generation,
            unsigned int individualsPerPopulation
        ) override;

        StretchFreeAlgorithm* buildStretchFreeAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            const vector<string>& parameterFileArray
        ) override;
};
