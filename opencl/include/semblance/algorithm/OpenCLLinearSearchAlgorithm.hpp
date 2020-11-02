#pragma once

#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"

#include <memory>
#include <unordered_map>

using namespace std;

class OpenCLLinearSearchAlgorithm : public LinearSearchAlgorithm {
    private:
        unordered_map<string, cl::Kernel> kernels;

    public:
        OpenCLLinearSearchAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder
        );

        void compileKernels(const string& deviceKernelSourcePath) override;
        void computeSemblanceAtGpuForMidpoint(float m0) override;
        void initializeParameters() override;
        void selectTracesToBeUsedForMidpoint(float m0) override;
};
