#define SPITS_ENTRY_POINT

#include "common/include/execution/SpitzFactoryAdapter.hpp"
#include "common/include/parser/DifferentialEvolutionParser.hpp"

#include <spits.hpp>

using namespace std;

spits::factory *spits_factory = new SpitzFactoryAdapter(
    DifferentialEvolutionParser::getInstance()
);
