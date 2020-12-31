#pragma once

#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLComputeAlgorithm.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

using namespace std;

class OpenCLLinearSearchAlgorithm : public LinearSearchAlgorithm, public OpenCLComputeAlgorithm {
    public:
        OpenCLLinearSearchAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder,
            unsigned int threadCount
        );

        void compileKernels(const string& deviceKernelSourcePath) override;
        void computeSemblanceAtGpuForMidpoint(float m0) override;
        void initializeParameters() override;
        void selectTracesToBeUsedForMidpoint(float m0) override;
};
