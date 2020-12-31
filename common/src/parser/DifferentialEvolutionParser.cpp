#include "common/include/output/Logger.hpp"
#include "common/include/parser/DifferentialEvolutionParser.hpp"

using namespace boost::program_options;
using namespace std;

unique_ptr<Parser> DifferentialEvolutionParser::instance = nullptr;

DifferentialEvolutionParser::DifferentialEvolutionParser() : Parser() {
    arguments.add_options()
        ("generations", value<unsigned int>()->required(), "Number of generations to be used in differential evolution.")
        ("population-size", value<unsigned int>()->required(), "Differential evolution population size.");
}

ComputeAlgorithm* DifferentialEvolutionParser::parseComputeAlgorithm(
    ComputeAlgorithmBuilder* builder,
    shared_ptr<DeviceContext> deviceContext,
    shared_ptr<Traveltime> traveltime
) const {
    unsigned int threadCount, generations, individualsPerPopulation;

    if (!argumentMap.count("generations") || !argumentMap.count("population-size")) {
        throw logic_error("Missing parameters for differential evolution.");
    }

    threadCount = argumentMap["thread-count"].as<unsigned int>();

    generations = argumentMap["generations"].as<unsigned int>();

    LOGD("Generations = " << generations);

    individualsPerPopulation = argumentMap["population-size"].as<unsigned int>();

    LOGD("Individuals per population = " << individualsPerPopulation);

    ComputeAlgorithm* algorithm = builder->buildDifferentialEvolutionAlgorithm(
        traveltime, deviceContext, threadCount, generations, individualsPerPopulation);

    if (argumentMap.count("kernel-path")) {
        algorithm->compileKernels(argumentMap["kernel-path"].as<string>());
    }

    return algorithm;
}

const string DifferentialEvolutionParser::getParserType() const {
    return "de";
}

Parser* DifferentialEvolutionParser::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<DifferentialEvolutionParser>();
    }
    return instance.get();
}
