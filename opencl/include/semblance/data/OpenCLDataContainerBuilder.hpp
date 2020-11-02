#pragma once

#include "common/include/semblance/data/DataContainerBuilder.hpp"

#include <memory>

using namespace std;

class OpenCLDataContainerBuilder : public DataContainerBuilder {
    protected:
        static unique_ptr<OpenCLDataContainerBuilder> instance;

    public:
        DataContainer* build(unsigned int allocatedCount, shared_ptr<DeviceContext> deviceContext) override;

        static DataContainerBuilder* getInstance();
};
