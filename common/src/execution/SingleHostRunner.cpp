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

void SingleHostRunner::workerThread(SingleHostRunner *ref, unsigned int deviceId) {
    LOGI("[" << deviceId << "] Worker thread start time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

    float m0;

    unique_ptr<ComputeAlgorithm> computeAlgorithm(ref->getComputeAlgorithm());

    mutex& resultSetMutex = ref->getResultSetMutex();
    mutex& queueMutex = ref->getQueueMutex();

    queue<float>& mipointQueue = ref->getMidpointQueue();

    ResultSet* resultSet = ref->getResultSet();

    LOGI("[" << deviceId << "] GPU Set up time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

    chrono::duration<double> setUpTime = chrono::duration<double>::zero();
    MEASURE_EXEC_TIME(setUpTime, computeAlgorithm->setUp());
    LOGI("Set up time = " << setUpTime.count() << " s");

    chrono::duration<double> mutexLockDuration = chrono::duration<double>::zero();
    chrono::duration<double> resultSetMutexLockDuration = chrono::duration<double>::zero();

    while (1) {

        chrono::steady_clock::time_point mutexLockTime = chrono::steady_clock::now();

        LOGI("------");
        LOGI("[" << deviceId << "] Mutex Lock Time Point = " << chrono::time_point_cast<chrono::milliseconds>(mutexLockTime).time_since_epoch().count());

        queueMutex.lock();

        if (mipointQueue.empty()) {
            queueMutex.unlock();
            break;
        }

        m0 = mipointQueue.front();
        mipointQueue.pop();

        queueMutex.unlock();

        mutexLockDuration += chrono::steady_clock::now() - mutexLockTime;

        LOGI("[" << deviceId << "][" << m0 << "] Compute Semblance Time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

        computeAlgorithm->computeSemblanceAndParametersForMidpoint(m0);

        chrono::steady_clock::time_point resultSetMutexLockTime = chrono::steady_clock::now();

        LOGI("[" << deviceId << "][" << m0 << "] Saving Results Time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

        resultSetMutex.lock();

        resultSet->setAllResultsForMidpoint(m0, computeAlgorithm->getComputedResults());

        for (unsigned int statResultIdx = 0; statResultIdx < static_cast<unsigned int>(StatisticResult::CNT); statResultIdx++) {
            StatisticResult statResult = static_cast<StatisticResult>(statResultIdx);
            resultSet->setStatisticalResultForMidpoint(m0, statResult, computeAlgorithm->getStatisticalResult(statResult));
        }

        resultSetMutex.unlock();

        resultSetMutexLockDuration += chrono::steady_clock::now() - resultSetMutexLockTime;
    }

    LOGI("[" << deviceId << "] End thread Time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

    LOGI("Queue mutex blocked time = " << mutexLockDuration.count() << " s");
    LOGI("Result set mutex blocked time = " << resultSetMutexLockDuration.count() << " s");
}

int SingleHostRunner::main(int argc, const char *argv[]) {

    try {
        chrono::steady_clock::time_point startTimePoint = chrono::steady_clock::now();

        LOGI("Start Time Point = " << chrono::time_point_cast<chrono::milliseconds>(startTimePoint).time_since_epoch().count());

        Gather* gather = Gather::getInstance();

        chrono::steady_clock::time_point numOfDevicesTimePoint = chrono::steady_clock::now();
        unsigned int devicesCount = parser->getNumberOfDevices();
        chrono::duration<double> totalNumOfDevicesTime = std::chrono::steady_clock::now() - startTimePoint;

        vector<thread> threads(devicesCount);

        LOGI("Parser Time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

        parser->parseArguments(argc, argv);

        traveltime.reset(parser->parseTraveltime());

        Dumper dumper(parser->getOutputDirectory(), parser->getFilename(), parser->getParserType(), traveltime->getTraveltimeWord());

        dumper.createDir();

        chrono::duration<double> totalReadTime = chrono::duration<double>::zero();
        MEASURE_EXEC_TIME(totalReadTime, parser->readGather());

        for (auto it : gather->getCdps()) {
            midpointQueue.push(it.first);
        }

        resultSet = make_unique<ResultSet>(traveltime->getNumberOfResults(), gather->getSamplesPerTrace());

        for(unsigned int deviceId = 0; deviceId < devicesCount; deviceId++) {
            threads[deviceId] = thread(workerThread, this, deviceId);
        }

        for(unsigned int deviceId = 0; deviceId < devicesCount; deviceId++) {
            threads[deviceId].join();
        }

        chrono::steady_clock::time_point writeTimePoint = chrono::steady_clock::now();

        LOGI("Dump Gather Time Point = " << chrono::time_point_cast<chrono::milliseconds>(writeTimePoint).time_since_epoch().count());

        dumper.dumpGatherParameters(parser->getInputFilePath());

        LOGI("Dump Traveltime Time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

        dumper.dumpTraveltime(traveltime.get());

        LOGI("Dump Result Time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

        for (unsigned int i = 0; i < traveltime->getNumberOfResults(); i++) {
            dumper.dumpResult(traveltime->getDescriptionForResult(i), resultSet->getArrayForResult(i));
        }

        LOGI("Dump Stat Time Point = " << chrono::time_point_cast<chrono::milliseconds>(chrono::steady_clock::now()).time_since_epoch().count());

        for (unsigned int i = 0; i < static_cast<unsigned int>(StatisticResult::CNT); i++) {
            StatisticResult statResult = static_cast<StatisticResult>(i);
            const StatisticalMidpointResult& statisticalResult = resultSet->get(statResult);
            const string& statResultName = STATISTIC_NAME_MAP[statResult];
            dumper.dumpStatisticalResult(statResultName, statisticalResult);
            LOGI("Average of " << statResultName << " is " << statisticalResult.getAverageOfAllMidpoints());
        }

        chrono::duration<double> totalWriteTime = std::chrono::steady_clock::now() - writeTimePoint;

        chrono::duration<double> totalExecutionTime = std::chrono::steady_clock::now() - startTimePoint;

        LOGI("Get Number of Devices time is " << totalNumOfDevicesTime.count() << " s");
        LOGI("Read time is " << totalReadTime.count() << " s");
        LOGI("Write time is " << totalWriteTime.count() << " s");
        LOGI("Results written to " << dumper.getOutputDirectoryPath());
        LOGI("It took " << totalExecutionTime.count() << "s to compute.");

        LOGI("End Time Point = " << chrono::time_point_cast<chrono::milliseconds>(std::chrono::steady_clock::now()).time_since_epoch().count());

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
