#pragma once

#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"

class DifferentialEvolutionAlgorithm : public ComputeAlgorithm {

    protected:
        unique_ptr<DataContainer> x, u, v, fx, fu, min, max, randomSeed;

        unsigned int generations, individualsPerPopulation;

    public:
        DifferentialEvolutionAlgorithm(
            shared_ptr<Traveltime> model,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder,
            unsigned int threadCount,
            unsigned int gen,
            unsigned int ind
        );

        void computeSemblanceAndParametersForMidpoint(float m0) override;
        void setUp() override;
        const string toString() const override;

        //
        // Virtual methods.
        //

        virtual void setupRandomSeedArray() = 0;
        virtual void startAllPopulations() = 0;
        virtual void mutateAllPopulations() = 0;
        virtual void crossoverPopulationIndividuals() = 0;
        virtual void advanceGeneration() = 0;
        virtual void selectBestIndividuals(vector<float>& resultArrays) = 0;
};
