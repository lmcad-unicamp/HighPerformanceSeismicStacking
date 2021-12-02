#pragma once

#include "common/include/parser/Parser.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithmBuilder.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <memory>
#include <mutex>
#include <spits.hpp>

using namespace std;

class SpitzWorker : public spits::worker {
    protected:
        spits::metrics& metrics;
        shared_ptr<ComputeAlgorithm> computeAlgorithm;
    public:
        SpitzWorker(shared_ptr<ComputeAlgorithm> computeAlgorithm, spits::metrics& metrics);
        int run(spits::istream& task, const spits::pusher& result);
};
