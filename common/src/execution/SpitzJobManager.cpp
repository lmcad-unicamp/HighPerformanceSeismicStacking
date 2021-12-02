#include "common/include/output/Logger.hpp"
#include "common/include/parser/Parser.hpp"
#include "common/include/execution/SpitzJobManager.hpp"

#include <iostream>

using namespace std;

SpitzJobManager::SpitzJobManager(shared_ptr<chrono::steady_clock::time_point>& timePoint
) : cdpIterator(Gather::getInstance()->getCdps().cbegin()),
    startTimePoint(timePoint) {
    LOGI("[JM] Job manager created.");
}

bool SpitzJobManager::next_task(const spits::pusher& task) {
    unique_lock<mutex> mlock(iteratorMutex);

    Gather* gather = Gather::getInstance();

    if (startTimePoint == nullptr) {
        LOGI("[JM] Job manager started sending tasks to workers.");
        startTimePoint.reset(new chrono::steady_clock::time_point(chrono::steady_clock::now()));
    }

    if (cdpIterator != gather->getCdps().end()) {

        spits::ostream taskStream;

        taskStream << cdpIterator->first;

        task.push(taskStream);

        cdpIterator++;

        return true;
    }

    return false;
}
