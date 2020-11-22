#ifndef SEMBL_ALGORITHM_LINEAR_SEARCH_H
#define SEMBL_ALGORITHM_LINEAR_SEARCH_H

#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/traveltime/Traveltime.hpp"

class LinearSearchAlgorithm : public ComputeAlgorithm {
    protected:
        unordered_map<SemblanceCommonResult, unique_ptr<DataContainer>> commonResultDeviceArrayMap;

        vector<unsigned int> discretizationGranularity, discretizationDivisor;

        vector<float> discretizationStep;

        unsigned int threadCountToRestore;

        void setupArrays();

        void setupDiscretizationSteps();

    public:
        LinearSearchAlgorithm(
            shared_ptr<Traveltime> model,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder
        );

        void computeSemblanceAndParametersForMidpoint(float m0) override;

        void setUp() override;

        const string toString() const override;

        float getParameterValueAt(unsigned int iterationNumber, unsigned int p) const;

        unsigned int getTotalNumberOfParameters() const;

        void setDiscretizationDivisorForParameter(unsigned int p, unsigned int d);

        void setDiscretizationGranularityForParameter(unsigned int p, unsigned int d);

        virtual void initializeParameters() = 0;
};
#endif
