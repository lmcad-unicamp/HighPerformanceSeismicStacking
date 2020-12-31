#pragma once

#include "common/include/semblance/algorithm/DifferentialEvolutionAlgorithm.hpp"

#include <curand_kernel.h>

class CudaDifferentialEvolutionAlgorithm : public DifferentialEvolutionAlgorithm {
    protected:
        curandState *st;

    public:
        CudaDifferentialEvolutionAlgorithm(
            shared_ptr<Traveltime> model,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder,
            unsigned int threadCount,
            unsigned int gen,
            unsigned int ind
        );

        ~CudaDifferentialEvolutionAlgorithm();

        void computeSemblanceAtGpuForMidpoint(float m0) override;

        void selectTracesToBeUsedForMidpoint(float m0) override;

        void setupRandomSeedArray() override;

        void startAllPopulations() override;

        void mutateAllPopulations() override;

        void crossoverPopulationIndividuals() override;

        void advanceGeneration() override;

        void selectBestIndividuals(vector<float>& resultArrays) override;
};
