#include "common/include/execution/Utils.hpp"
#include "common/include/model/Trace.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/semblance/algorithm/StretchFreeAlgorithm.hpp"

#include <fstream>
#include <sstream>

#define NMAX 10

using namespace std;

StretchFreeAlgorithm::StretchFreeAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int threadCount,
    const vector<string>& files
) : ComputeAlgorithm("strecth-free", model, context, dataBuilder, threadCount),
    parameterFileArray(files) {
}

void StretchFreeAlgorithm::computeSemblanceAndParametersForMidpoint(float m0) {
    LOGI("Computing semblance for m0 = " << m0);

    Gather* gather = Gather::getInstance();

    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();
    unsigned int numberOfSamplesPerTrace = gather->getSamplesPerTrace();

    deviceContext->activate();

    deviceResultArray->reset();
    deviceNotUsedCountArray->reset();

    chrono::duration<double> selectionExecutionTime = chrono::duration<double>::zero();
    chrono::duration<double> totalExecutionTime = chrono::duration<double>::zero();

    MEASURE_EXEC_TIME(selectionExecutionTime, selectTracesToBeUsedForMidpoint(m0));

    MEASURE_EXEC_TIME(totalExecutionTime, computeSemblanceAtGpuForMidpoint(m0));

    deviceResultArray->pasteTo(computedResults);

    float totalUsedTracesCount = static_cast<float>(filteredTracesCount) *
            static_cast<float>(numberOfSamplesPerTrace) *
            static_cast<float>(totalNumberOfParameters);

    saveStatisticalResults(totalUsedTracesCount, totalExecutionTime, selectionExecutionTime);
}

unsigned int StretchFreeAlgorithm::getTotalNumberOfParameters() const {
    /* TODO: improve this */
    return 2 * NMAX + 1;
}

void StretchFreeAlgorithm::setUp() {
    Gather* gather = Gather::getInstance();

    unsigned int numberOfResults = traveltime->getNumberOfCommonResults() + 1;
    unsigned int numberOfSamples = gather->getSamplesPerTrace();
    unsigned int totalNumberOfParameters = getTotalNumberOfParameters();

    copyGatherDataToDevice();

    deviceParameterArray.reset(dataFactory->build(totalNumberOfParameters, deviceContext));
    deviceNotUsedCountArray.reset(dataFactory->build(numberOfSamples * totalNumberOfParameters, deviceContext));
    deviceResultArray.reset(dataFactory->build(numberOfResults * numberOfSamples, deviceContext));

    computedResults.resize(numberOfResults * numberOfSamples);

    commonResultDeviceArrayMap[SemblanceCommonResult::SEMBL].reset(dataFactory->build(numberOfSamples * totalNumberOfParameters, deviceContext));
    commonResultDeviceArrayMap[SemblanceCommonResult::STACK].reset(dataFactory->build(numberOfSamples * totalNumberOfParameters, deviceContext));

    readNonStretchedFreeParameterFromFile();

    vector<float> nArray(totalNumberOfParameters);
    unsigned int idx = 0;
    for (int n = -NMAX; n <= NMAX; n++, idx++) {
        nArray[idx] = static_cast<float>(n);
    }
    deviceParameterArray->copyFrom(nArray);
}

const string StretchFreeAlgorithm::toString() const {

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    ostringstream stringStream;

    stringStream << "Total number of parameters = " << getTotalNumberOfParameters() << endl;
    stringStream << "N in [ " << -NMAX << "," << NMAX << "]" << endl;

    stringStream << "Parameter files: " << endl;
    for (unsigned int i = 0; i < numberOfParameters; i++) {
         stringStream << parameterFileArray[i] << endl;
    }

    return stringStream.str();
}

void StretchFreeAlgorithm::readNonStretchedFreeParameterFromFile() {
    Gather* gather = Gather::getInstance();

    unsigned int numberOfParameters = traveltime->getNumberOfParameters();
    unsigned int numberOfSamples = gather->getSamplesPerTrace();

    vector<ifstream> fileArray(numberOfParameters);

    for(unsigned int i = 0; i < numberOfParameters; i++) {
        fileArray[i].open(parameterFileArray[i], ios::binary);
    }

    for (auto it = gather->getCdps().begin(); it != gather->getCdps().end(); it++) {

        float m0 = it->first;

        nonStretchFreeParameters[m0].reset(dataFactory->build(numberOfParameters * numberOfSamples, deviceContext));

        for(unsigned int i = 0; i < numberOfParameters; i++) {
            Trace t;
            t.read(fileArray[i], gather->getAzimuthInRad());
            nonStretchFreeParameters[m0]->copyFromWithOffset(t.getSamples(),  i * numberOfSamples);
        }
    }
}
