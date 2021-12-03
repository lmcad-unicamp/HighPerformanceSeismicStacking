#pragma once

#include "common/include/model/Cdp.hpp"
#include "common/include/model/Gather.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <spits.hpp>

using namespace std;

class SpitzJobManager : public spits::job_manager {
    private:
        bool isStartTimePointSet;
        map<float, Cdp>::const_iterator cdpIterator;
        mutex iteratorMutex;
        shared_ptr<chrono::steady_clock::time_point> startTimePoint;

    public:
        SpitzJobManager(shared_ptr<chrono::steady_clock::time_point> timePoint);

        bool next_task(const spits::pusher& task) override;
};
