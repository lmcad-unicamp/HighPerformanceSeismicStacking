#pragma once

#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"

#include <memory>

class CudaLinearSearchAlgorithm : public LinearSearchAlgorithm {

    public:
        CudaLinearSearchAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder
        );

        void computeSemblanceAtGpuForMidpoint(float m0) override;

        void initializeParameters() override;

        void selectTracesToBeUsedForMidpoint(float m0) override;
};
