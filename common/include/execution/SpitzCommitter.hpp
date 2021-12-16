#pragma once

#include "common/include/model/Gather.hpp"
#include "common/include/output/Dumper.hpp"
#include "common/include/semblance/result/ResultSet.hpp"

#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <spits.hpp>
#include <string>
#include <vector>

using namespace std;

class SpitzCommitter : public spits::committer {
    private:
        mutex taskMutex;

        shared_ptr<Traveltime> traveltime;

        shared_ptr<chrono::steady_clock::time_point> startTimePoint;
        chrono::steady_clock::time_point committerStartTimePoint;

        string filePath;

        Dumper dumper;

        unique_ptr<ResultSet> resultSet;

        unsigned int taskCount, taskIndex;

        vector<float> tempResultArray;

    public:
        SpitzCommitter(
            shared_ptr<Traveltime> traveltime,
            shared_ptr<chrono::steady_clock::time_point> timePoint,
            const string& folder,
            const string& file,
            const string& computeMethod
        );

        int commit_task(spits::istream& result);
        int commit_job(const spits::pusher& final_result);
};
