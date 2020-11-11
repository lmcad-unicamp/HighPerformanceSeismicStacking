#pragma once

#include "common/include/semblance/algorithm/StretchFreeAlgorithm.hpp"

#include <memory>

class OpenCLStretchFreeAlgorithm : public StretchFreeAlgorithm {
    public:
        OpenCLStretchFreeAlgorithm(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder,
            const vector<string>& files
        );

        void compileKernels(const string& deviceKernelSourcePath) override;
        void computeSemblanceAtGpuForMidpoint(float m0) override;
        void selectTracesToBeUsedForMidpoint(float m0) override;
};