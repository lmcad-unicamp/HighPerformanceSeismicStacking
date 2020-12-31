#include "common/include/output/Logger.hpp"
#include "cuda/include/execution/CudaUtils.hpp"
#include "cuda/include/semblance/data/CudaDeviceContext.hpp"

#include <cuda.h>
#include <sstream>
#include <stdexcept>

using namespace std;

CudaDeviceContext::CudaDeviceContext(unsigned int devId) : DeviceContext(devId) {
}

void CudaDeviceContext::activate() const {
    LOGH("Activating GPU#" << deviceId);
    CUDA_ASSERT(cudaSetDevice(static_cast<int>(deviceId)));
}
