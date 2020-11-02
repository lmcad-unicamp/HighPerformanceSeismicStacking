#pragma once

#include "common/include/semblance/data/DeviceContext.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"

#include <memory>
#include <string>

using namespace std;

#define OPENCL_CONTEXT_PTR(context_ptr) dynamic_cast<OpenCLDeviceContext*>(context_ptr.get())

class OpenCLDeviceContext : public DeviceContext { 

    private:
        unique_ptr<cl::Platform> platform;
        unique_ptr<cl::Device> device;
        unique_ptr<cl::Context> context;
        unique_ptr<cl::CommandQueue> commandQueue;

    public:
        OpenCLDeviceContext(unsigned int deviceId);

        cl::Context& getContext();
        cl::CommandQueue& getCommandQueue();
        cl::Device& getDevice();

        void activate() const override;
};
