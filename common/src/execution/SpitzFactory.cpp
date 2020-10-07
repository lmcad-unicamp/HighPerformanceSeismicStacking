#include "common/include/execution/SpitzFactory.hpp"
#include "common/include/model/Gather.hpp"
#include "common/include/output/Logger.hpp"

#include <memory>
#include <spits.hpp>

using namespace std;

SpitzFactory::SpitzFactory(
    Parser* p,
    ComputeAlgorithmBuilder* builder,
    DeviceContextBuilder* deviceBuilder
) : parser(p), builder(builder), deviceBuilder(deviceBuilder) {
}

void SpitzFactory::initialize(int argc, const char *argv[]) {
    Gather* gather = Gather::getInstance();

    parser->parseArguments(argc, argv);

    if (!gather->isGatherRead()) {
        parser->readGather();
    }

    if (traveltime == nullptr) {
        LOGI("Initializing traveltime");
        traveltime.reset(parser->parseTraveltime());
    }

    LOGI("Factory initialized");
}

spits::job_manager *SpitzFactory::create_job_manager(
    int argc,
    const char *argv[],
    spits::istream& jobinfo,
    spits::metrics& metrics
) {
    initialize(argc, argv);
    return new SpitzJobManager();
}

spits::committer *SpitzFactory::create_committer(
    int argc,
    const char *argv[],
    spits::istream& jobinfo,
    spits::metrics& metrics
) {
    initialize(argc, argv);
    return new SpitzCommitter(traveltime, parser->getOutputDirectory(), parser->getFilename());
}

spits::worker *SpitzFactory::create_worker(
    int argc,
    const char *argv[],
    spits::metrics& metrics
) {
    unique_lock<mutex> mlock(deviceMutex);

    initialize(argc, argv);

    LOGI("Device count is " << deviceCount);

    shared_ptr<DeviceContext> deviceContext(deviceBuilder->build(deviceCount++));

    ComputeAlgorithm* computeAlgorithm = parser->parseComputeAlgorithm(builder, deviceContext, traveltime);

    return new SpitzWorker(computeAlgorithm, metrics);
}
