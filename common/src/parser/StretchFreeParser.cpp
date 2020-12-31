#include "common/include/parser/StretchFreeParser.hpp"
#include "common/include/traveltime/StretchFreeTraveltimeWrapper.hpp"

using namespace boost::program_options;
using namespace std;

unique_ptr<Parser> StretchFreeParser::instance = nullptr;

StretchFreeParser::StretchFreeParser() : Parser() {
    arguments.add_options()
        ("stack-datafiles", value<vector<string>>()->required()->multitoken(), "Datafiles to be used for stretch-free stack generation.");
}

ComputeAlgorithm* StretchFreeParser::parseComputeAlgorithm(
    ComputeAlgorithmBuilder* builder,
    shared_ptr<DeviceContext> deviceContext,
    shared_ptr<Traveltime> traveltime
) const {
    vector<string> nonStretchFreeParameterFiles;

    if (argumentMap.count("stack-datafiles")) {
        nonStretchFreeParameterFiles = argumentMap["stack-datafiles"].as<vector<string>>();
                LOGI(nonStretchFreeParameterFiles.size());
    }

    unsigned int threadCount = argumentMap["thread-count"].as<unsigned int>();

    ComputeAlgorithm* algorithm = builder->buildStretchFreeAlgorithm(
        traveltime, deviceContext, threadCount, nonStretchFreeParameterFiles);

    if (argumentMap.count("kernel-path")) {
        algorithm->compileKernels(argumentMap["kernel-path"].as<string>());
    }

    return algorithm;
}

Traveltime* StretchFreeParser::parseTraveltime() const {
    return new StretchFreeTraveltimeWrapper(Parser::parseTraveltime());
}

const string StretchFreeParser::getParserType() const {
    return "stretch_free";
}

Parser* StretchFreeParser::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<StretchFreeParser>();
    }
    return instance.get();
}
