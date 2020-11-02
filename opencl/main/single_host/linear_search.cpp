#include "common/include/parser/LinearSearchParser.hpp"
#include "opencl/include/execution/OpenCLSingleHostRunner.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    OpenCLSingleHostRunner singleHostExecution(LinearSearchParser::getInstance());
    return singleHostExecution.main(argc, const_cast<const char**>(argv));
}
