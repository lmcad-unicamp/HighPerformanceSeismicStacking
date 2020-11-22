#define SPITS_ENTRY_POINT

#include "common/include/execution/SpitzFactory.hpp"
#include "common/include/parser/DifferentialEvolutionParser.hpp"
#include "opencl/include/semblance/algorithm/OpenCLComputeAlgorithmBuilder.hpp"
#include "opencl/include/semblance/data/OpenCLDeviceContextBuilder.hpp"

#include <spits.hpp>

using namespace std;

spits::factory *spits_factory = new SpitzFactory(
    DifferentialEvolutionParser::getInstance(),
    OpenCLComputeAlgorithmBuilder::getInstance(),
    OpenCLDeviceContextBuilder::getInstance()
);
