#include "common/include/output/Logger.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/data/OpenCLDataContainer.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <sstream>
#include <stdexcept>

using namespace std;

OpenCLDataContainer::OpenCLDataContainer(
    unsigned int elementCount,
    shared_ptr<DeviceContext> context
) : DataContainer(elementCount, context) {
    allocate();
}

cl::Buffer& OpenCLDataContainer::getBuffer() const {
    return *openClBuffer;
}

void OpenCLDataContainer::allocate() {
    cl_int errorCode;

    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    openClBuffer.reset(new cl::Buffer(
        openClContext->getContext(),
        CL_MEM_READ_ONLY,
        elementCount * sizeof(float),
        &errorCode
    ));

    OPENCL_ASSERT_CODE(errorCode);

    reset();
}

void OpenCLDataContainer::copyFrom(const vector<float>& sourceArray) {
    if (elementCount < sourceArray.size()) {
        throw invalid_argument("Allocated memory in GPU is different from source array size.");
    }

    copyFromWithOffset(sourceArray, 0);
}

void OpenCLDataContainer::copyFromWithOffset(const vector<float>& sourceArray, unsigned int offset) {
    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    cl::CommandQueue& commandQueue = openClContext->getCommandQueue();

    OPENCL_ASSERT(commandQueue.enqueueWriteBuffer(
        getBuffer(),
        CL_TRUE,
        offset * sizeof(float),
        sourceArray.size() * sizeof(float),
        sourceArray.data()
    ));
}

void OpenCLDataContainer::deallocate() {
    openClBuffer.reset();
}

void OpenCLDataContainer::pasteTo(vector<float>& targetArray) {
    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    cl::CommandQueue& commandQueue = openClContext->getCommandQueue();

    if (elementCount > targetArray.size()) {
        throw invalid_argument("Allocated memory in GPU is different from target array size.");
    }

    OPENCL_ASSERT(commandQueue.enqueueReadBuffer(
        getBuffer(),
        CL_TRUE,
        0,
        targetArray.size() * sizeof(float),
        targetArray.data()
    ));
}

void OpenCLDataContainer::reset() {
    auto openClContext = OPENCL_CONTEXT_PTR(deviceContext);

    cl::CommandQueue& commandQueue = openClContext->getCommandQueue();

    OPENCL_ASSERT(commandQueue.enqueueFillBuffer<float>(
        getBuffer(),
        0,
        0,
        elementCount * sizeof(float)
    ));
}
