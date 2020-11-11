#pragma once

#include "common/include/semblance/data/DeviceContext.hpp"
#include "common/include/traveltime/Traveltime.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class OpenCLComputeAlgorithm {
    protected:
        unordered_map<string, cl::Kernel> kernels;

    public:
        void compileKernels(
            const string& deviceKernelSourcePath, 
            const string& computeAlgorithm, 
            shared_ptr<Traveltime> traveltime,
            shared_ptr<DeviceContext> context
        );
        vector<string> readSourceFiles(const vector<string>& files) const;
};
