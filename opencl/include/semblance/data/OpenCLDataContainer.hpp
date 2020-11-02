#pragma once

#include "common/include/semblance/data/DataContainer.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"

#include <memory>

#define OPENCL_DEV_BUFFER(_ptr) dynamic_cast<OpenCLDataContainer*>(_ptr.get())->getBuffer()

using namespace std;

class OpenCLDataContainer : public DataContainer {

    private:
        unique_ptr<cl::Buffer> openClBuffer;

    public:
        OpenCLDataContainer(unsigned int elementCount, shared_ptr<DeviceContext> context);

        cl::Buffer& getBuffer() const;

        void allocate() override;

        void copyFrom(const std::vector<float>& sourceArray) override;

        void copyFromWithOffset(const std::vector<float>& sourceArray, unsigned int offset) override;

        void deallocate() override;

        void pasteTo(std::vector<float>& targetArray) override;

        void reset() override;
};
