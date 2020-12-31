#include "common/include/execution/SpitzFactory.hpp"
#include "common/include/execution/SpitzFactoryAdapter.hpp"
#include "common/include/model/Gather.hpp"
#include "common/include/output/Logger.hpp"

#include <memory>
#include <spits.hpp>

using namespace std;

SpitzFactory::SpitzFactory(
    Parser* p,
    ComputeAlgorithmBuilder* builder,
    DeviceContextBuilder* deviceBuilder
) : SpitzFactoryAdapter(p), builder(builder), deviceBuilder(deviceBuilder) {
}

spits::worker *SpitzFactory::create_worker(
    int argc,
    const char *argv[],
    spits::metrics& metrics
) {
    unique_lock<mutex> mlock(deviceMutex);

    initialize(argc, argv);

    LOGD("Device count is " << deviceCount);

    shared_ptr<DeviceContext> deviceContext(deviceBuilder->build(deviceCount++));

    ComputeAlgorithm* computeAlgorithm = parser->parseComputeAlgorithm(builder, deviceContext, traveltime);

    return new SpitzWorker(computeAlgorithm, metrics);
}
