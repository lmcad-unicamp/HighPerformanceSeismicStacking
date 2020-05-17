#ifndef CUDA_SEMBL_DATA_FACTORY_H
#define CUDA_SEMBL_DATA_FACTORY_H

#include "common/include/semblance/data/DataContainerBuilder.hpp"

#include <memory>

using namespace std;

class CudaDataContainerBuilder : public DataContainerBuilder {
    protected:
        static unique_ptr<CudaDataContainerBuilder> instance;

    public:
        DataContainer* build(unsigned int allocatedCount, shared_ptr<DeviceContext> deviceContext) override;

        static DataContainerBuilder* getInstance();
};
#endif
