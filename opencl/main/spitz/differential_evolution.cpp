#define SPITS_ENTRY_POINT

#include "common/include/execution/SpitzFactory.hpp"
#include "common/include/parser/DifferentialEvolutionParser.hpp"
#include "cuda/include/semblance/algorithm/CudaComputeAlgorithmBuilder.hpp"
#include "cuda/include/semblance/data/CudaDeviceContextBuilder.hpp"

#include <memory>
#include <spits.hpp>

using namespace std;

spits::factory *spits_factory = new SpitzFactory(
    DifferentialEvolutionParser::getInstance(),
    CudaComputeAlgorithmBuilder::getInstance(),
    CudaDeviceContextBuilder::getInstance()
);
