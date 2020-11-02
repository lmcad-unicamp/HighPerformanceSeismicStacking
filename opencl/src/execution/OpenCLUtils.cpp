#include "common/include/output/Logger.hpp"
#include "opencl/include/execution/OpenCLUtils.hpp"

#include <sstream>
#include <stdexcept>

void openClAssert(cl_int errorCode, const char *file, int line) {
    if (errorCode != CL_SUCCESS) {
        ostringstream stringStream;
        stringStream << "OpenCL error detected at " << file << "::" << line << " with error " << errorCode;
        LOGE(stringStream.str());
        throw runtime_error(stringStream.str());
    }
}
