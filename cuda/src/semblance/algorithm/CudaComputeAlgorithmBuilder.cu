#include "cuda/include/semblance/algorithm/CudaComputeAlgorithmBuilder.hpp"
#include "cuda/include/semblance/algorithm/CudaLinearSearchAlgorithm.hpp"
#include "cuda/include/semblance/algorithm/CudaDifferentialEvolutionAlgorithm.hpp"
#include "cuda/include/semblance/algorithm/CudaStretchFreeAlgorithm.hpp"
#include "cuda/include/semblance/data/CudaDataContainerBuilder.hpp"

#include <sstream>

using namespace std;

unique_ptr<ComputeAlgorithmBuilder> CudaComputeAlgorithmBuilder::instance = nullptr;

ComputeAlgorithmBuilder* CudaComputeAlgorithmBuilder::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<CudaComputeAlgorithmBuilder>();
    }
    return instance.get();
}

LinearSearchAlgorithm* CudaComputeAlgorithmBuilder::buildLinearSearchAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    const vector<int>& discretizationArray
) {
    if (discretizationArray.size() != traveltime->getNumberOfParameters()) {
        ostringstream exceptionString;
        exceptionString << traveltime->getTraveltimeWord() << " requires exact ";
        exceptionString << traveltime->getNumberOfParameters() << " discretization granularities.";
        throw logic_error(exceptionString.str());
    }

    DataContainerBuilder* dataFactory = CudaDataContainerBuilder::getInstance();

    LinearSearchAlgorithm* computeAlgorithm = new CudaLinearSearchAlgorithm(traveltime, context, dataFactory);

    for (unsigned int i = 0; i < traveltime->getNumberOfParameters(); i++) {
        computeAlgorithm->setDiscretizationGranularityForParameter(i, discretizationArray[i]);
    }

    return computeAlgorithm;
}

DifferentialEvolutionAlgorithm* CudaComputeAlgorithmBuilder::buildDifferentialEvolutionAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    unsigned int generation,
    unsigned int individualsPerPopulation
) {
    DataContainerBuilder* dataFactory = CudaDataContainerBuilder::getInstance();

    return new CudaDifferentialEvolutionAlgorithm(traveltime, context, dataFactory, generation, individualsPerPopulation);
}

StretchFreeAlgorithm* CudaComputeAlgorithmBuilder::buildStretchFreeAlgorithm(
    shared_ptr<Traveltime> traveltime,
    shared_ptr<DeviceContext> context,
    const vector<string>& parameterFileArray
) {
    DataContainerBuilder* dataFactory = CudaDataContainerBuilder::getInstance();

    if (parameterFileArray.size() != traveltime->getNumberOfParameters()) {
        ostringstream exceptionString;
        exceptionString << traveltime->getTraveltimeWord() << " requires exact ";
        exceptionString << traveltime->getNumberOfParameters() << " parameter files.";
        throw logic_error(exceptionString.str());
    }

    return new CudaStretchFreeAlgorithm(traveltime, context, dataFactory, parameterFileArray);
}
