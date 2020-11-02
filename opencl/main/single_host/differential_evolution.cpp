#include "common/include/parser/DifferentialEvolutionParser.hpp"
#include "opencl/include/execution/OpenCLSingleHostRunner.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    OpenCLSingleHostRunner singleHostExecution(DifferentialEvolutionParser::getInstance());
    return singleHostExecution.main(argc, const_cast<const char**>(argv));
}
