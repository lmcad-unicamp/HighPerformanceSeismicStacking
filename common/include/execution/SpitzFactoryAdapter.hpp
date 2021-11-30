#pragma once

#include "common/include/execution/SpitzCommitter.hpp"
#include "common/include/execution/SpitzJobManager.hpp"
#include "common/include/execution/SpitzWorker.hpp"
#include "common/include/parser/Parser.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <spits.hpp>

using namespace std;

class SpitzFactoryAdapter : public spits::factory {
    protected:
        Parser* parser;
        shared_ptr<chrono::steady_clock::time_point> startTimePoint;
        shared_ptr<Traveltime> traveltime;

    public:
        SpitzFactoryAdapter(Parser* p);

        spits::job_manager *create_job_manager(
            int argc,
            const char *argv[],
            spits::istream& jobinfo,
            spits::metrics& metrics
        ) override;

        spits::committer *create_committer(
            int argc,
            const char *argv[],
            spits::istream& jobinfo,
            spits::metrics& metrics
        ) override;

        virtual spits::worker *create_worker(
            int argc,
            const char *argv[],
            spits::metrics& metrics
        ) override;

        void initialize(int argc, const char *argv[]);
};
