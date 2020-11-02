#ifndef OPENCL_SINGLE_HOST_H
#define OPENCL_SINGLE_HOST_H

#include "common/include/execution/SingleHostRunner.hpp"

using namespace std;

class OpenCLSingleHostRunner : public SingleHostRunner {

    public:
        OpenCLSingleHostRunner(Parser* parser);

        unsigned int getNumOfDevices() const override;
};
#endif
