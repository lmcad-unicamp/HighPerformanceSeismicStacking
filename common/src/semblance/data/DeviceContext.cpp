#include "common/include/semblance/data/DeviceContext.hpp"

DeviceContext::DeviceContext(unsigned int devId) : deviceId(devId) {
}

unsigned int DeviceContext::getDeviceId() const {
    return deviceId;
}

DeviceContext::~DeviceContext() {
}
