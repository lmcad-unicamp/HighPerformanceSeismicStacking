#pragma once

#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/cl2.hpp>
#include <string>

using namespace std;

#define OPENCL_ASSERT(ans) do { openClAssert((ans), __FILE__, __LINE__); } while(0);
#define OPENCL_ASSERT_CODE(_errorCode) do { openClAssert(_errorCode, __FILE__, __LINE__); } while(0);

unsigned int fitGlobal(unsigned int global, unsigned int threadCount);
void openClAssert(cl_int errorCode, const char *file, int line);
const string openClErrorMessage(cl_int errorCode);
