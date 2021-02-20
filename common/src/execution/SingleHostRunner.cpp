#include "common/include/execution/SingleHostRunner.hpp"
#include "common/include/execution/Utils.hpp"
#include "common/include/output/Dumper.hpp"

#include <cerrno>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std;

SingleHostRunner::SingleHostRunner(
    Parser* p,
    ComputeAlgorithmBuilder* algorithmBuilder,
    DeviceContextBuilder* deviceContextBuilder
) : parser(p),
    algorithmBuilder(algorithmBuilder),
    deviceContextBuilder(deviceContextBuilder),
    deviceIndex(0) {
}

ResultSet* SingleHostRunner::getResultSet() {
    return resultSet.get();
}

queue<float>& SingleHostRunner::getMidpointQueue(unsigned int threadId) {
    return midpointQueues.at(threadId);
}

mutex& SingleHostRunner::getDeviceMutex() {
    return deviceMutex;
}

mutex& SingleHostRunner::getResultSetMutex() {
    return resultSetMutex;
}

ComputeAlgorithm* SingleHostRunner::getComputeAlgorithm() {
    lock_guard<mutex> autoLock(deviceMutex);
    shared_ptr<DeviceContext> devContext(deviceContextBuilder->build(deviceIndex++));

    ComputeAlgorithm* computeAlgorithm = parser->parseComputeAlgorithm(algorithmBuilder, devContext, traveltime);

    chrono::duration<double> setUpTime = chrono::duration<double>::zero();
    MEASURE_EXEC_TIME(setUpTime, computeAlgorithm->setUp());
    LOGI("Set up time = " << setUpTime.count() << " s");

    return computeAlgorithm;
}

void SingleHostRunner::workerThread(SingleHostRunner *ref, unsigned int threadIndex) {
    float m0;

    unique_ptr<ComputeAlgorithm> computeAlgorithm(ref->getComputeAlgorithm());

    mutex& resultSetMutex = ref->getResultSetMutex();
    queue<float>& midpointQueue = ref->getMidpointQueue(threadIndex);
    ResultSet* resultSet = ref->getResultSet();

    chrono::duration<double> resultSetMutexLockDuration = chrono::duration<double>::zero();

    while (!midpointQueue.empty()) {

        m0 = midpointQueue.front();
        midpointQueue.pop();

        computeAlgorithm->computeSemblanceAndParametersForMidpoint(m0);

        chrono::steady_clock::time_point resultSetMutexLockTime = chrono::steady_clock::now();

        resultSetMutex.lock();

        resultSet->setAllResultsForMidpoint(m0, computeAlgorithm->getComputedResults());

        for (unsigned int statResultIdx = 0; statResultIdx < static_cast<unsigned int>(StatisticResult::CNT); statResultIdx++) {
            StatisticResult statResult = static_cast<StatisticResult>(statResultIdx);
            resultSet->setStatisticalResultForMidpoint(m0, statResult, computeAlgorithm->getStatisticalResult(statResult));
        }

        resultSetMutex.unlock();

        resultSetMutexLockDuration += chrono::steady_clock::now() - resultSetMutexLockTime;
    }

    LOGI("Result set mutex blocked time = " << resultSetMutexLockDuration.count() << " s");
}

int SingleHostRunner::main(int argc, const char *argv[]) {

    try {
        chrono::steady_clock::time_point startTimePoint = chrono::steady_clock::now();

        Gather* gather = Gather::getInstance();

        unsigned int devicesCount = getNumOfDevices();

        vector<thread> threads(devicesCount);

        parser->parseArguments(argc, argv);

        traveltime.reset(parser->parseTraveltime());

        Dumper dumper(parser->getOutputDirectory(), parser->getFilename(), parser->getParserType(), traveltime->getTraveltimeWord());

        dumper.createDir();

        chrono::duration<double> totalReadTime = chrono::duration<double>::zero();
        MEASURE_EXEC_TIME(totalReadTime, parser->readGather());

        unsigned int itIndex = 0;
        midpointQueues.resize(devicesCount);
        for (auto it : gather->getCdps()) {
            unsigned int queueIndex = itIndex % devicesCount;
            midpointQueues[queueIndex].push(it.first);
            itIndex++;
        }

        resultSet = make_unique<ResultSet>(traveltime->getNumberOfResults(), gather->getSamplesPerTrace());

        for(unsigned int deviceId = 0; deviceId < devicesCount; deviceId++) {
            threads[deviceId] = thread(workerThread, this, deviceId);
        }

        for(unsigned int deviceId = 0; deviceId < devicesCount; deviceId++) {
            threads[deviceId].join();
        }

        chrono::steady_clock::time_point writeTimePoint = chrono::steady_clock::now();

        dumper.dumpGatherParameters(parser->getInputFilePath());

        dumper.dumpTraveltime(traveltime.get());

        for (unsigned int i = 0; i < traveltime->getNumberOfResults(); i++) {
            dumper.dumpResult(traveltime->getDescriptionForResult(i), resultSet->getArrayForResult(i));
        }

        for (unsigned int i = 0; i < static_cast<unsigned int>(StatisticResult::CNT); i++) {
            StatisticResult statResult = static_cast<StatisticResult>(i);
            const StatisticalMidpointResult& statisticalResult = resultSet->get(statResult);
            const string& statResultName = STATISTIC_NAME_MAP[statResult];
            dumper.dumpStatisticalResult(statResultName, statisticalResult);
            LOGI("Average of " << statResultName << " is " << statisticalResult.getAverageOfAllMidpoints());
        }

        chrono::duration<double> totalWriteTime = std::chrono::steady_clock::now() - writeTimePoint;

        chrono::duration<double> totalExecutionTime = std::chrono::steady_clock::now() - startTimePoint;

        LOGI("Read time is " << totalReadTime.count() << " s");
        LOGI("Write time is " << totalWriteTime.count() << " s");
        LOGI("Results written to " << dumper.getOutputDirectoryPath());
        LOGI("It took " << totalExecutionTime.count() << "s to compute.");

        return 0;
    }
    catch (const invalid_argument& e) {
        cout << e.what() << endl;
        parser->printHelp();
        return 0;
    }
    catch (const runtime_error& e) {
        cout << e.what() << endl;
        return -ENODEV;
    }
    catch (const exception& e) {
        cout << e.what() << endl;
        return -1;
    }
}
