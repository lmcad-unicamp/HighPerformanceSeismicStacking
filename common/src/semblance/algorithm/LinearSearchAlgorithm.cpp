#include "common/include/execution/Utils.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/semblance/algorithm/LinearSearchAlgorithm.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>

using namespace std;

LinearSearchAlgorithm::LinearSearchAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : ComputeAlgorithm("linear-search", model, context, dataBuilder),
    discretizationGranularity(model->getNumberOfParameters()),
    discretizationDivisor(model->getNumberOfParameters()),
    discretizationStep(model->getNumberOfParameters()) {
}

void LinearSearchAlgorithm::computeSemblanceAndParametersForMidpoint(float m0) {
    Gather* gather = Gather::getInstance();

    unsigned int numberOfSamplesPerTrace = gather->getSamplesPerTrace();
    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    deviceContext->activate();

    deviceResultArray->reset();
    deviceNotUsedCountArray->reset();
    commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL]->reset();
    commonResultDeviceArrayMap[SemblanceCommonResult::STACK]->reset();

    chrono::duration<double> selectionExecutionTime = chrono::duration<double>::zero();
    chrono::duration<double> totalExecutionTime = chrono::duration<double>::zero();

    MEASURE_EXEC_TIME(selectionExecutionTime, selectTracesToBeUsedForMidpoint(m0));

    LOGI("totalNumberOfParameters = " << totalNumberOfParameters);

    MEASURE_EXEC_TIME(totalExecutionTime, computeSemblanceAtGpuForMidpoint(m0));

    deviceResultArray->pasteTo(computedResults);

    unsigned long totalUsedTracesCount;
    totalUsedTracesCount = static_cast<unsigned long>(filteredTracesCount) *
            static_cast<unsigned long>(numberOfSamplesPerTrace) *
            static_cast<unsigned long>(totalNumberOfParameters);

    saveStatisticalResults(totalUsedTracesCount, totalExecutionTime, selectionExecutionTime);
}

float LinearSearchAlgorithm::getParameterValueAt(unsigned int iterationNumber, unsigned int p) const {
    unsigned int step = (iterationNumber / discretizationDivisor[p]) % discretizationGranularity[p];
    return traveltime->getLowerBoundForParameter(p) + static_cast<float>(step) * discretizationStep[p];
}

unsigned int LinearSearchAlgorithm::getTotalNumberOfParameters() const {
    return accumulate(
            discretizationGranularity.begin(),
            discretizationGranularity.end(),
            1, multiplies<unsigned int>()
        );
}

void LinearSearchAlgorithm::setDiscretizationDivisorForParameter(unsigned int p, unsigned int d) {
    if (p >= traveltime->getNumberOfParameters()) {
        throw invalid_argument("Parameter index is out of bounds");
    }
    discretizationDivisor[p] = d;
}

void LinearSearchAlgorithm::setDiscretizationGranularityForParameter(
    unsigned int parameterIndex,
    unsigned int granularity) {

    if (parameterIndex >= traveltime->getNumberOfParameters()) {
        throw invalid_argument("Parameter index is out of bounds");
    }

    discretizationGranularity[parameterIndex] = granularity;
}

void LinearSearchAlgorithm::setUp() {

    chrono::duration<double> initalizationExecutionTime = chrono::duration<double>::zero();

    deviceContext->activate();

    setupArrays();

    copyGatherDataToDevice();

    setupDiscretizationSteps();

    MEASURE_EXEC_TIME(initalizationExecutionTime, initializeParameters());

    LOGI("initalizationExecutionTime = " << initalizationExecutionTime.count() << "s");

    isSet = true;
}

unsigned int LinearSearchAlgorithm::getParameterArrayStep() const {
    return min(getTotalNumberOfParameters(), threadCount);
}

void LinearSearchAlgorithm::setupArrays() {
    ostringstream stringStream;

    Gather* gather = Gather::getInstance();

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();
    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfResults = traveltime->getNumberOfResults();
    unsigned int numberOfSamples = gather->getSamplesPerTrace();

    deviceParameterArray.reset(dataFactory->build(numberOfParameters * totalNumberOfParameters, deviceContext));

    deviceNotUsedCountArray.reset(dataFactory->build(numberOfSamples * totalNumberOfParameters, deviceContext));

    commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL].reset(dataFactory->build(numberOfSamples * totalNumberOfParameters, deviceContext));
    commonResultDeviceArrayMap[SemblanceCommonResult::STACK].reset(dataFactory->build(numberOfSamples * totalNumberOfParameters, deviceContext));

    deviceResultArray.reset(dataFactory->build(numberOfResults * numberOfSamples, deviceContext));

    computedResults.resize(numberOfResults * numberOfSamples);
}

void LinearSearchAlgorithm::setupDiscretizationSteps() {
    unsigned int divisor = 1;
    for (unsigned int i = 0; i < traveltime->getNumberOfParameters(); i++) {

        float lowerBound = traveltime->getLowerBoundForParameter(i);
        float upperBound = traveltime->getUpperBoundForParameter(i);

        discretizationDivisor[i] = divisor;
        divisor *= discretizationGranularity[i];

        discretizationStep[i] = (upperBound - lowerBound) / static_cast<float>(discretizationGranularity[i]);
    }
}

const string LinearSearchAlgorithm::toString() const {
    ostringstream stringStream;

    stringStream << "Total number of parameters = " << getTotalNumberOfParameters() << endl;

    for (unsigned int i = 0; i < traveltime->getNumberOfParameters(); i++) {
        stringStream << "# of discrete values for " << traveltime->getDescriptionForParameter(i);
        stringStream << " = " << discretizationGranularity[i] << endl;
    }

    stringStream << "Thread count = " << threadCount << endl;

    return stringStream.str();
}
