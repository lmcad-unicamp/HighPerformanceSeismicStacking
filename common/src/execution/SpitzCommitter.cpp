#include "common/include/parser/Parser.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/execution/SpitzCommitter.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <chrono>
#include <memory>
#include <sstream>

using namespace std;

SpitzCommitter::SpitzCommitter(
    shared_ptr<Traveltime> model,
    shared_ptr<chrono::steady_clock::time_point> timePoint,
    const string& folder,
    const string& file,
    const string& computeMethod
) : traveltime(model),
    startTimePoint(timePoint),
    committerStartTimePoint(chrono::steady_clock::now()),
    filePath(file),
    dumper(folder, file, computeMethod, traveltime->getTraveltimeWord()) {

    Gather* gather = Gather::getInstance();

    unsigned int sampleCount = gather->getSamplesPerTrace();

    tempResultArray.resize(traveltime->getNumberOfResults() * sampleCount);

    resultSet = make_unique<ResultSet>(traveltime->getNumberOfResults(), gather->getSamplesPerTrace());

    taskCount = gather->getTotalCdpsCount();
    taskIndex = 0;

    dumper.createDir();

    LOGI("[CO] Committer created.");
}

int SpitzCommitter::commit_task(spits::istream& result) {

    unique_lock<mutex> mlock(taskMutex);

    float m0 = result.read_float();

    LOGI("Committing result for m0 = "<< m0);

    result.read_data(tempResultArray.data(), tempResultArray.size() * sizeof(float));

    resultSet->setAllResultsForMidpoint(m0, tempResultArray);

    for (unsigned int i = 0; i < static_cast<unsigned int>(StatisticResult::CNT); i++) {
        StatisticResult statResult = static_cast<StatisticResult>(i);
        resultSet->setStatisticalResultForMidpoint(m0, statResult, result.read_float());
    }

    taskIndex++;

    LOGI("[CO] Result committed. [" << taskIndex << "," << taskCount << "]");

    return 0;
}

int SpitzCommitter::commit_job(const spits::pusher& final_result) {

    dumper.dumpGatherParameters(filePath);

    for (unsigned int i = 0; i < traveltime->getNumberOfResults(); i++) {
        dumper.dumpResult(
            traveltime->getDescriptionForResult(i),
            resultSet->getArrayForResult(i)
        );
    }

    for (unsigned int i = 0; i < static_cast<unsigned int>(StatisticResult::CNT); i++) {
        StatisticResult statResult = static_cast<StatisticResult>(i);
        const StatisticalMidpointResult& statisticalResult = resultSet->get(statResult);
        const string& statResultName = STATISTIC_NAME_MAP[statResult];
        LOGI("[CO] Average of " << statResultName << " is " << statisticalResult.getAverageOfAllMidpoints());
        dumper.dumpStatisticalResult(statResultName, statisticalResult);
    }

    LOGI("[CO] Results written to " << dumper.getOutputDirectoryPath());

    chrono::steady_clock::time_point commitJobTimePoint(std::chrono::steady_clock::now());

    if (startTimePoint) {
        chrono::duration<double> totalExecutionTime = commitJobTimePoint - *startTimePoint;
        LOGI("[CO] Job completed. It took " << totalExecutionTime.count() << "s since first task started");
    }

    chrono::duration<double> totalExecutionTimeCommitter = commitJobTimePoint - committerStartTimePoint;
    LOGI("[CO] Job completed. It took " << totalExecutionTimeCommitter.count() << "s since committer started");

    // A result must be pushed even if the final result is not passed on
    final_result.push(NULL, 0);

    return 0;
}
