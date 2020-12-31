#pragma once

#include "common/include/semblance/algorithm/DifferentialEvolutionAlgorithm.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLComputeAlgorithm.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace std;

class OpenCLDifferentialEvolutionAlgorithm : public DifferentialEvolutionAlgorithm, public OpenCLComputeAlgorithm {
    protected:
        unique_ptr<cl::Buffer> st;

    public:
        OpenCLDifferentialEvolutionAlgorithm(
            shared_ptr<Traveltime> model,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder,
            unsigned int threadCount,
            unsigned int gen,
            unsigned int ind
        );

        void compileKernels(const string& deviceKernelSourcePath) override;

        void computeSemblanceAtGpuForMidpoint(float m0) override;

        void selectTracesToBeUsedForMidpoint(float m0) override;

        void setupRandomSeedArray() override;

        void startAllPopulations() override;

        void mutateAllPopulations() override;

        void crossoverPopulationIndividuals() override;

        void advanceGeneration() override;

        void selectBestIndividuals(vector<float>& resultArrays) override;
};
