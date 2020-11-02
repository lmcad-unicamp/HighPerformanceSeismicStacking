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
        context = make_unique<cl::Context>(*device);
        commandQueue = make_unique<cl::CommandQueue>(*context, *device);
    }

    throw runtime_error("Couldn't find device.");
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

// void OpenCLDeviceContext::compile(std::string pth) {

//     FILE *fp = fopen(pth.append("/semblance.cl").c_str(), "r");

//     uint32_t max_sz = 2 * 1024 * 1024;
//     cl_int ret;

//     char *source = (char*) calloc(max_sz, sizeof(char));
//     size_t source_sz = fread(source, sizeof(char), max_sz, fp);
//     fclose(fp);

//     prgm = clCreateProgramWithSource(ctx, 1, const_cast<const char**>(&source), &source_sz, &ret);

//     ret = clBuildProgram(prgm, 1, &device_id, "-cl-fast-relaxed-math", NULL, NULL);

//     if(ret != CL_SUCCESS) {
//         size_t len = 0;

//         clGetProgramBuildInfo(prgm, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

//         char *buffer = (char*) calloc(len, sizeof(char));

//         ret = clGetProgramBuildInfo(prgm, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);

//         LOGI("%s", buffer);

//         free(buffer);

//         throw "OpenCL program failed to build!";
//     }

//     sembl_kernel = clCreateKernel(prgm, "compute_semblance_gpu", &ret);
//     sembl_ga_kernel = clCreateKernel(prgm, "compute_semblance_ga_gpu", &ret);
//     stack_kernel = clCreateKernel(prgm, "compute_strech_free_sembl_gpu", &ret);
//     sembl_ft_cmp_crs = clCreateKernel(prgm, "search_for_traces_cmp_crs", &ret);
//     sembl_ft_crp = clCreateKernel(prgm, "search_for_traces_crp", &ret);

//     if(ret != CL_SUCCESS){
//         LOGI("clCreateKernel failed");
//         throw "OpenCL clCreateKernel failed!";
//     }

//     free(source);
// }
