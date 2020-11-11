#include "opencl/include/semblance/algorithm/OpenCLComputeAlgorithm.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContext.hpp"

#include <filesystem>
#include <fstream>

using namespace std;

void OpenCLComputeAlgorithm::compileKernels(
    const string& deviceKernelSourcePath,
    const string& computeAlgorithm,
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context
) {
    cl_int errorCode;

    vector<string> kernelSources;
    vector<cl::Kernel> openClKernels;

    auto openClContext = OPENCL_CONTEXT_PTR(context);

    const string computeAlgorithmKernelFile = computeAlgorithm + ".cl";
    const string traveltimeKernelPath = deviceKernelSourcePath + "/" + traveltime->getTraveltimeWord();

    filesystem::path commonKernelPath(deviceKernelSourcePath + "/common/" + computeAlgorithmKernelFile);
    filesystem::path specificKernelPath(traveltimeKernelPath + "/" + computeAlgorithmKernelFile);
    filesystem::path commonTraveltimeKernelPath(traveltimeKernelPath + "/common.cl");

    if (filesystem::exists(commonKernelPath)) {
        kernelSources.push_back(static_cast<string>(commonKernelPath));
    }

    if (filesystem::exists(specificKernelPath)) {
        kernelSources.push_back(static_cast<string>(specificKernelPath));
    }

    if (filesystem::exists(commonTraveltimeKernelPath)) {
        kernelSources.push_back(static_cast<string>(commonTraveltimeKernelPath));
    }

    cl::Program program(openClContext->getContext(), readSourceFiles(kernelSources), &errorCode);

    OPENCL_ASSERT_CODE(errorCode);

    errorCode = program.build({ openClContext->getDevice() }, "-cl-fast-relaxed-math -I../");

    if (errorCode != CL_SUCCESS) {
        auto errors = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(openClContext->getDevice(), &errorCode);
        ostringstream stringStream;
        stringStream << "Building cl::Program failed with " << errors;
        throw runtime_error(stringStream.str());
    }

    OPENCL_ASSERT(program.createKernels(&openClKernels));

    for(auto kernel : openClKernels) {
        string kernelName = kernel.getInfo<CL_KERNEL_FUNCTION_NAME>(&errorCode);
        OPENCL_ASSERT_CODE(errorCode);
        kernels[kernelName] = kernel;
    }
}

vector<string> OpenCLComputeAlgorithm::readSourceFiles(const vector<string>& files) const {
    vector<string> result;
    for (auto file : files) {
        ifstream kernelFile(file);
        stringstream kernelSourceCode;
        kernelSourceCode << kernelFile.rdbuf();
        result.push_back(kernelSourceCode.str());
    }
    return result;
}