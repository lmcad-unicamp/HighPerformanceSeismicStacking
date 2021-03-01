#include "common/include/execution/SingleHostRunner.hpp"
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

queue<float>& SingleHostRunner::getMidpointQueue() {
    return midpointQueue;
}

mutex& SingleHostRunner::getDeviceMutex() {
    return deviceMutex;
}

mutex& SingleHostRunner::getResultSetMutex() {
    return resultSetMutex;
}

mutex& SingleHostRunner::getQueueMutex() {
    return queueMutex;
}

ComputeAlgorithm* SingleHostRunner::getComputeAlgorithm() {
    lock_guard<mutex> autoLock(deviceMutex);
    shared_ptr<DeviceContext> devContext(deviceContextBuilder->build(deviceIndex++));
    return parser->parseComputeAlgorithm(algorithmBuilder, devContext, traveltime);
}

void SingleHostRunner::workerThread(SingleHostRunner *ref) {
    float m0;

    unique_ptr<ComputeAlgorithm> computeAlgorithm(ref->getComputeAlgorithm());

    mutex& resultSetMutex = ref->getResultSetMutex();
    mutex& queueMutex = ref->getQueueMutex();

    queue<float>& mipointQueue = ref->getMidpointQueue();

    ResultSet* resultSet = ref->getResultSet();

    computeAlgorithm->setUp();

    while (1) {

        queueMutex.lock();

        if (mipointQueue.empty()) {
            queueMutex.unlock();
            break;
        }

        m0 = mipointQueue.front();
        mipointQueue.pop();

        queueMutex.unlock();

        computeAlgorithm->computeSemblanceAndParametersForMidpoint(m0);

        resultSetMutex.lock();

        resultSet->setAllResultsForMidpoint(m0, computeAlgorithm->getComputedResults());

        for (unsigned int statResultIdx = 0; statResultIdx < static_cast<unsigned int>(StatisticResult::CNT); statResultIdx++) {
            StatisticResult statResult = static_cast<StatisticResult>(statResultIdx);
            resultSet->setStatisticalResultForMidpoint(m0, statResult, computeAlgorithm->getStatisticalResult(statResult));
        }

        resultSetMutex.unlock();
    }
}

int SingleHostRunner::main(int argc, const char *argv[]) {

    try {
        chrono::steady_clock::time_point startTimePoint = chrono::steady_clock::now();

        Gather* gather = Gather::getInstance();

        chrono::steady_clock::time_point initDriverTimePoint = chrono::steady_clock::now();
        unsigned int devicesCount = getNumOfDevices();
        chrono::duration<double> initDriverDuration = std::chrono::steady_clock::now() - initDriverTimePoint;

        vector<thread> threads(devicesCount);

        chrono::duration<double> parseArgsDuration = chrono::duration<double>::zero();
        MEASURE_EXEC_TIME(parseArgsDuration, parser->parseArguments(argc, argv));

        traveltime.reset(parser->parseTraveltime());

        Dumper dumper(parser->getOutputDirectory(), parser->getFilename(), parser->getParserType(), traveltime->getTraveltimeWord());

        dumper.createDir();

        chrono::duration<double> readGatherDuration = chrono::duration<double>::zero();
        MEASURE_EXEC_TIME(readGatherDuration, parser->readGather());

        for (auto it : gather->getCdps()) {
            midpointQueue.push(it.first);
        }

        resultSet = make_unique<ResultSet>(traveltime->getNumberOfResults(), gather->getSamplesPerTrace());

        LOGI("Starting worker threads...");
        chrono::steady_clock::time_point startWorkerThreadTimePoint = chrono::steady_clock::now();

        for(unsigned int deviceId = 0; deviceId < devicesCount; deviceId++) {
            threads[deviceId] = thread(workerThread, this);
        }

        for(unsigned int deviceId = 0; deviceId < devicesCount; deviceId++) {
            threads[deviceId].join();
        }

        chrono::duration<double> workerThreadDuration = std::chrono::steady_clock::now() - startWorkerThreadTimePoint;
        chrono::steady_clock::time_point dumpResultsTimePoint = chrono::steady_clock::now();

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

        chrono::duration<double> dumpResultsDuration = std::chrono::steady_clock::now() - dumpResultsTimePoint;
        chrono::duration<double> totalExecutionTime = std::chrono::steady_clock::now() - startTimePoint;

        LOGI("Driver init. duration is " << initDriverDuration.count() << " s");
        LOGI("Read gather duration is " << readGatherDuration.count() << " s");
        LOGI("Parse args. duration is " << parseArgsDuration.count() << " s");
        LOGI("Worker threads duration is " << workerThreadDuration.count() << " s");
        LOGI("Dump results duration is " << dumpResultsDuration.count() << " s");
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
