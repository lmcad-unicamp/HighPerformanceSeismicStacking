#define SPITS_ENTRY_POINT

#include "common/include/execution/SpitzFactoryAdapter.hpp"
#include "common/include/parser/LinearSearchParser.hpp"

#include <memory>
#include <spits.hpp>

using namespace std;

spits::factory *spits_factory = new SpitzFactoryAdapter(
    LinearSearchParser::getInstance()
);
