#include "opencl/include/semblance/algorithm/OpenCLComputeAlgorithmBuilder.hpp"
#include "opencl/include/semblance/algorithm/OpenCLLinearSearchAlgorithm.hpp"
#include "opencl/include/semblance/algorithm/OpenCLDifferentialEvolutionAlgorithm.hpp"
#include "opencl/include/semblance/algorithm/OpenCLStretchFreeAlgorithm.hpp"
#include "opencl/include/semblance/data/OpenCLDataContainerBuilder.hpp"

#include <memory>
#include <sstream>

using namespace std;

unique_ptr<ComputeAlgorithmBuilder> OpenCLComputeAlgorithmBuilder::instance = nullptr;

ComputeAlgorithmBuilder* OpenCLComputeAlgorithmBuilder::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<OpenCLComputeAlgorithmBuilder>();
    }
    return instance.get();
}

LinearSearchAlgorithm* OpenCLComputeAlgorithmBuilder::buildLinearSearchAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    unsigned int threadCount,
    const vector<int>& discretizationArray
) {
    if (discretizationArray.size() != traveltime->getNumberOfParameters()) {
        ostringstream exceptionString;
        exceptionString << traveltime->getTraveltimeWord() << " requires exact ";
        exceptionString << traveltime->getNumberOfParameters() << " discretization granularities.";
        throw logic_error(exceptionString.str());
    }

    DataContainerBuilder* dataFactory = OpenCLDataContainerBuilder::getInstance();

    LinearSearchAlgorithm* computeAlgorithm = new OpenCLLinearSearchAlgorithm(traveltime, context, dataFactory, threadCount);

    for (unsigned int i = 0; i < traveltime->getNumberOfParameters(); i++) {
        computeAlgorithm->setDiscretizationGranularityForParameter(i, discretizationArray[i]);
    }

    return computeAlgorithm;
}

DifferentialEvolutionAlgorithm* OpenCLComputeAlgorithmBuilder::buildDifferentialEvolutionAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    unsigned int threadCount,
    unsigned int generation,
    unsigned int individualsPerPopulation
) {
    DataContainerBuilder* dataFactory = OpenCLDataContainerBuilder::getInstance();

    return new OpenCLDifferentialEvolutionAlgorithm(traveltime, context, dataFactory, threadCount, generation, individualsPerPopulation);
}

StretchFreeAlgorithm* OpenCLComputeAlgorithmBuilder::buildStretchFreeAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    unsigned int threadCount,
    const vector<string>& parameterFileArray
) {
    DataContainerBuilder* dataFactory = OpenCLDataContainerBuilder::getInstance();

    if (parameterFileArray.size() != traveltime->getNumberOfParameters()) {
        ostringstream exceptionString;
        exceptionString << traveltime->getTraveltimeWord() << " requires exact ";
        exceptionString << traveltime->getNumberOfParameters() << " parameter files.";
        throw logic_error(exceptionString.str());
    }

    return new OpenCLStretchFreeAlgorithm(traveltime, context, dataFactory, threadCount, parameterFileArray);
}
