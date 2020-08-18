#include "common/include/parser/Parser.hpp"
#include "common/include/output/Dumper.hpp"
#include "common/include/output/Logger.hpp"
#include "common/include/execution/SpitzCommitter.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <chrono>
#include <memory>
#include <sstream>

using namespace std;

SpitzCommitter::SpitzCommitter(
    shared_ptr<Traveltime> model,
    const string& folder,
    const string& file
) : traveltime(model), folderPath(folder), filePath(file) {

    Gather* gather = Gather::getInstance();

    unsigned int sampleCount = gather->getSamplesPerTrace();

    tempResultArray.resize(traveltime->getNumberOfResults() * sampleCount);

    resultSet = make_unique<ResultSet>(traveltime->getNumberOfResults(), gather->getSamplesPerTrace());

    taskCount = gather->getTotalCdpsCount();
    taskIndex = 0;

    startTimePoint = chrono::steady_clock::now();

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

    Dumper dumper(folderPath, filePath);

    dumper.createDir();

    dumper.dumpGatherParameters(filePath);

    for (unsigned int i = 0; i < traveltime->getNumberOfResults(); i++) {
        dumper.dumpResult(
            traveltime->getDescriptionForResult(i),
            resultSet->getArrayForResult(i)
        );
    }

    for (unsigned int i = 0; i < static_cast<unsigned int>(StatisticResult::CNT); i++) {
        StatisticResult statResult = static_cast<StatisticResult>(i);
        dumper.dumpStatisticalResult(STATISTIC_NAME_MAP[statResult], resultSet->get(statResult));
    }

    // A result must be pushed even if the final result is not passed on
    final_result.push(NULL, 0);

    chrono::duration<double> totalExecutionTime = std::chrono::steady_clock::now() - startTimePoint;

    LOGI("[CO] Job completed. It took " << totalExecutionTime.count() << "s");

    return 0;
}
