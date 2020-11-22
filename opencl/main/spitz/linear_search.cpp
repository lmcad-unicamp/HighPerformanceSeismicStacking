#define SPITS_ENTRY_POINT

#include "common/include/execution/SpitzFactory.hpp"
#include "common/include/parser/LinearSearchParser.hpp"
#include "opencl/include/semblance/algorithm/OpenCLComputeAlgorithmBuilder.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContextBuilder.hpp"

#include <memory>
#include <spits.hpp>

using namespace std;

spits::factory *spits_factory = new SpitzFactory(
    LinearSearchParser::getInstance(),
    OpenCLComputeAlgorithmBuilder::getInstance(),
    OpenCLDeviceContextBuilder::getInstance()
);
