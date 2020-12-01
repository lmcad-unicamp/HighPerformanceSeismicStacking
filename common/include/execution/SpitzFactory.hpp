#pragma once

#include "common/include/execution/SpitzCommitter.hpp"
#include "common/include/execution/SpitzFactoryAdapter.hpp"
#include "common/include/execution/SpitzJobManager.hpp"
#include "common/include/execution/SpitzWorker.hpp"
#include "common/include/parser/Parser.hpp"
#include "common/include/semblance/data/DeviceContextBuilder.hpp"

#include <memory>
#include <mutex>
#include <spits.hpp>

using namespace std;

class SpitzFactory : public SpitzFactoryAdapter {
    protected:
        mutex deviceMutex;

        ComputeAlgorithmBuilder* builder;

        DeviceContextBuilder* deviceBuilder;

        unsigned int deviceCount = 0;

    public:
        SpitzFactory(Parser* p, ComputeAlgorithmBuilder* builder, DeviceContextBuilder* deviceBuilder);

        spits::worker *create_worker(
            int argc,
            const char *argv[],
            spits::metrics& metrics
        ) override;
};
