#include "common/include/output/Logger.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace std;

OpenCLDeviceContext::OpenCLDeviceContext(unsigned int deviceId) : DeviceContext(deviceId) {
    vector<cl::Platform> platforms;
    vector<cl::Device> devicesPerPlatform;

    OPENCL_ASSERT(cl::Platform::get(&platforms));

    unsigned int numberOfPlatforms = static_cast<unsigned int>(platforms.size());

    int deviceCount = static_cast<int>(deviceId);
    bool foundDevice = false;
    for (unsigned int platformId = 0; platformId < numberOfPlatforms && !foundDevice; platformId++) {

        OPENCL_ASSERT(platforms[platformId].getDevices(CL_DEVICE_TYPE_GPU, &devicesPerPlatform));

        int devicePerPlatformCount = static_cast<int>(devicesPerPlatform.size());

        if (deviceCount - devicePerPlatformCount > 0) {
            deviceCount -= devicePerPlatformCount;
            continue;
        }

        foundDevice = true;
        platform = make_unique<cl::Platform>(platforms[platformId]);
        device = make_unique<cl::Device>(devicesPerPlatform[deviceCount]);
    }

    if (foundDevice) {
        cl_int errorCode;

        context = make_unique<cl::Context>(*device, nullptr, nullptr, nullptr, &errorCode);
        OPENCL_ASSERT_CODE(errorCode);

        commandQueue = make_unique<cl::CommandQueue>(*context, *device, CL_QUEUE_PROFILING_ENABLE, &errorCode);
        OPENCL_ASSERT_CODE(errorCode);
    }
    else {
        throw runtime_error("Couldn't find device.");
    }
}

cl::Context& OpenCLDeviceContext::getContext() {
    return *context;
}

cl::CommandQueue& OpenCLDeviceContext::getCommandQueue() {
    return *commandQueue;
}

cl::Device& OpenCLDeviceContext::getDevice() {
    return *device;
}

void OpenCLDeviceContext::activate() const {
}
