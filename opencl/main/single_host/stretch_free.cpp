#include "common/include/parser/StretchFreeParser.hpp"
#include "opencl/include/execution/OpenCLSingleHostRunner.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    OpenCLSingleHostRunner singleHostExecution(StretchFreeParser::getInstance());
    return singleHostExecution.main(argc, const_cast<const char**>(argv));
}
