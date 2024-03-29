#include "common/include/execution/SpitzFactoryAdapter.hpp"
#include "common/include/model/Gather.hpp"
#include "common/include/output/Logger.hpp"

#include <memory>
#include <spits.hpp>
#include <stdexcept>

using namespace std;

SpitzFactoryAdapter::SpitzFactoryAdapter(
    Parser* p
) : parser(p), startTimePoint(make_shared<chrono::steady_clock::time_point>()) {
}

void SpitzFactoryAdapter::initialize(int argc, const char *argv[]) {
    Gather* gather = Gather::getInstance();

    parser->parseArguments(argc, argv);

    if (!gather->isGatherRead()) {
        parser->readGather();
    }

    if (traveltime == nullptr) {
        LOGD("Initializing traveltime");
        traveltime.reset(parser->parseTraveltime());
    }

    LOGD("Factory initialized");
}

spits::job_manager *SpitzFactoryAdapter::create_job_manager(
    int argc,
    const char *argv[],
    spits::istream& jobinfo,
    spits::metrics& metrics
) {
    initialize(argc, argv);
    return new SpitzJobManager(startTimePoint);
}

spits::committer *SpitzFactoryAdapter::create_committer(
    int argc,
    const char *argv[],
    spits::istream& jobinfo,
    spits::metrics& metrics
) {
    initialize(argc, argv);
    return new SpitzCommitter(traveltime, startTimePoint, parser->getOutputDirectory(), parser->getFilename(), parser->getParserType());
}

spits::worker *SpitzFactoryAdapter::create_worker(
    int argc,
    const char *argv[],
    spits::metrics& metrics
) {
    throw logic_error("Create worker method is not implemented. Provide a device-specific implementation.");
}
