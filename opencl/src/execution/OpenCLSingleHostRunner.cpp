#include "opencl/include/execution/OpenCLSingleHostRunner.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLComputeAlgorithmBuilder.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContextBuilder.hpp"

OpenCLSingleHostRunner::OpenCLSingleHostRunner(Parser* parser) : 
    SingleHostRunner(parser, OpenCLComputeAlgorithmBuilder::getInstance(), OpenCLDeviceContextBuilder::getInstance()) {
}

unsigned int OpenCLSingleHostRunner::getNumOfDevices() const {
    vector<cl::Platform> platforms;
    vector<cl::Device> devicesPerPlatform;

    unsigned int numberOfDevices = 0;

    OPENCL_ASSERT(cl::Platform::get(&platforms));

    unsigned int numberOfPlatforms = static_cast<unsigned int>(platforms.size());

    for (unsigned int platformId = 0; platformId < numberOfPlatforms; platformId++) {
        OPENCL_ASSERT(platforms[platformId].getDevices(CL_DEVICE_TYPE_GPU, &devicesPerPlatform));
        numberOfDevices += static_cast<unsigned int>(devicesPerPlatform.size());
    }

    return numberOfDevices;
}
