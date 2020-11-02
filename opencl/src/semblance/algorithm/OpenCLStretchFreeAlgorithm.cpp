#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLStretchFreeAlgorithm.hpp"

OpenCLStretchFreeAlgorithm::OpenCLStretchFreeAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    const vector<string>& files
) : StretchFreeAlgorithm(traveltime, context, dataBuilder, files) {
}

void OpenCLStretchFreeAlgorithm::compileKernels(const string& deviceKernelSourcePath) {

}

void OpenCLStretchFreeAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {

}

void OpenCLStretchFreeAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {

}
