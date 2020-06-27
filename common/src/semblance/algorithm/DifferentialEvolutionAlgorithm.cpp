#include "common/include/execution/Utils.hpp"
#include "common/include/semblance/algorithm/DifferentialEvolutionAlgorithm.hpp"

#include <sstream>

using namespace std;

DifferentialEvolutionAlgorithm::DifferentialEvolutionAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int gen,
    unsigned int ind
) : ComputeAlgorithm("de", model, context, dataBuilder),
    generations(gen),
    individualsPerPopulation(ind) {
}

void DifferentialEvolutionAlgorithm::computeSemblanceAndParametersForMidpoint(float m0) {
    Gather* gather = Gather::getInstance();

    unsigned int numberOfSamplesPerTrace = gather->getSamplesPerTrace();

    chrono::duration<double> selectionExecutionTime = chrono::duration<double>::zero();
    chrono::duration<double> totalExecutionTime = chrono::duration<double>::zero();

    deviceNotUsedCountArray->reset();
    fx->reset();
    fu->reset();

    /* Initialize population randomly */
    startAllPopulations();

    deviceParameterArray.swap(x);
    deviceResultArray.swap(fx);

    /* Copy samples, midpoints and halfoffsets to the device */
    MEASURE_EXEC_TIME(selectionExecutionTime, selectTracesToBeUsedForMidpoint(m0));

    /* Compute fitness for initial population */
    MEASURE_EXEC_TIME(totalExecutionTime, computeSemblanceAtGpuForMidpoint(m0));

    deviceParameterArray.swap(x);
    deviceResultArray.swap(fx);

    for (unsigned gen = 1; gen < generations; gen++) {

        /* Mutation */
        mutateAllPopulations();

        /* Crossover */
        crossoverPopulationIndividuals();

        deviceParameterArray.swap(u);
        deviceResultArray.swap(fu);

        MEASURE_EXEC_TIME(totalExecutionTime, computeSemblanceAtGpuForMidpoint(m0));

        deviceParameterArray.swap(u);
        deviceResultArray.swap(fu);

        /* Selection */
        advanceGeneration();
    }

    selectBestIndividuals(computedResults);

    unsigned long totalUsedTracesCount;
    totalUsedTracesCount = static_cast<unsigned long>(filteredTracesCount) *
            static_cast<unsigned long>(numberOfSamplesPerTrace) *
            static_cast<unsigned long>(generations) *
            static_cast<unsigned long>(individualsPerPopulation);

    saveStatisticalResults(totalUsedTracesCount, totalExecutionTime, selectionExecutionTime);
}

unsigned int DifferentialEvolutionAlgorithm::getParameterArrayStep() const {
    return individualsPerPopulation * traveltime->getNumberOfParameters();
}

unsigned int DifferentialEvolutionAlgorithm::getResultArrayStep() const {
    return individualsPerPopulation * traveltime->getNumberOfCommonResults();
};

void DifferentialEvolutionAlgorithm::setUp() {
    Gather* gather = Gather::getInstance();

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfResults = traveltime->getNumberOfResults();
    unsigned int numberOfSamples = gather->getSamplesPerTrace();
    unsigned int parameterArrayStep = getParameterArrayStep();
    unsigned int resultArrayStep = getResultArrayStep();

    unsigned int parameterArraySize = gather->getSamplesPerTrace() * parameterArrayStep;
    unsigned int commonResultArraySize = gather->getSamplesPerTrace() * resultArrayStep;

    copyGatherDataToDevice();

    vector<float> lowerBounds(numberOfParameters), upperBounds(numberOfParameters);

    for (unsigned int i = 0; i < numberOfParameters; i++) {
        lowerBounds[i] = traveltime->getLowerBoundForParameter(i);
        upperBounds[i] = traveltime->getUpperBoundForParameter(i);
    }

    min.reset(dataFactory->build(numberOfParameters, deviceContext));
    max.reset(dataFactory->build(numberOfParameters, deviceContext));

    min->copyFrom(lowerBounds);
    max->copyFrom(upperBounds);

    deviceNotUsedCountArray.reset(dataFactory->build(gather->getSamplesPerTrace() * individualsPerPopulation, deviceContext));

    x.reset(dataFactory->build(parameterArraySize, deviceContext));
    u.reset(dataFactory->build(parameterArraySize, deviceContext));
    v.reset(dataFactory->build(parameterArraySize, deviceContext));

    fx.reset(dataFactory->build(commonResultArraySize, deviceContext));
    fu.reset(dataFactory->build(commonResultArraySize, deviceContext));

    computedResults.resize(numberOfResults * numberOfSamples);

    setupRandomSeedArray();
}

const string DifferentialEvolutionAlgorithm::toString() const {
    ostringstream stringStream;

    stringStream << "Total number of generations = " << generations << endl;
    stringStream << "Total number of ind. per population = " << individualsPerPopulation << endl;

    return stringStream.str();
}
