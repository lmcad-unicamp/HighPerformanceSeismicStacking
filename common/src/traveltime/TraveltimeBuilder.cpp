#include "common/include/output/Logger.hpp"
#include "common/include/traveltime/TraveltimeBuilder.hpp"
#include "common/include/traveltime/CommonMidPoint.hpp"
#include "common/include/traveltime/OffsetContinuationTrajectory.hpp"
#include "common/include/traveltime/CommonReflectionSurface.hpp"

#include <math.h>
#include <memory>
#include <sstream>
#include <stdexcept>

using namespace std;

unique_ptr<TraveltimeBuilder> TraveltimeBuilder::instance = nullptr;

TraveltimeBuilder* TraveltimeBuilder::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<TraveltimeBuilder>();
    }
    return instance.get();
}

Traveltime* TraveltimeBuilder::buildCommonMidPoint(
    const vector<float>& lowerBounds,
    const vector<float>& upperBounds
) {

    Traveltime* traveltime = new CommonMidPoint();
    unsigned int modelParameterCount = traveltime->getNumberOfParameters();

    if (lowerBounds.size() != modelParameterCount || upperBounds.size() != modelParameterCount) {
        ostringstream exceptionString;
        exceptionString << "CMP requires exact " << modelParameterCount << " lower and upper bounds.";
        throw logic_error(exceptionString.str());
    }

    for (unsigned int i = 0; i < modelParameterCount; i++) {
        traveltime->updateLowerBoundForParameter(i, lowerBounds[i]);
        traveltime->updateUpperBoundForParameter(i, upperBounds[i]);
    }

    return traveltime;
}

Traveltime* TraveltimeBuilder::buildZeroOffsetCommonReflectionSurface(
    const vector<float>& lowerBounds, const vector<float>& upperBounds,
    float v0, float aIn, float bPctg) {

    CommonReflectionSurface* traveltime = new CommonReflectionSurface();

    if (lowerBounds.size() < 1 || upperBounds.size() < 1) {
        throw logic_error("ZOCRS requires at least a single lower/upper bound.");
    }

    traveltime->updateLowerBoundForParameter(0, lowerBounds[0]);
    traveltime->updateUpperBoundForParameter(0, upperBounds[0]);

    traveltime->setReferenceVelocity(v0);
    traveltime->setAinInDegrees(aIn);
    traveltime->setRatioForB(bPctg);

    float minVelocity = traveltime->getLowerBoundForParameter(0);
    float upperBoundForA = 2.0f * sin(traveltime->getAinInRad()) / v0;
    float upperBoundForB = bPctg * 4 / (minVelocity  * minVelocity);

    LOGD("Upper bound for A is " << upperBoundForA);
    LOGD("Upper bound for B is " << upperBoundForB);

    traveltime->updateLowerBoundForParameter(1, -upperBoundForA);
    traveltime->updateUpperBoundForParameter(1, upperBoundForA);

    traveltime->updateLowerBoundForParameter(2, -upperBoundForB);
    traveltime->updateUpperBoundForParameter(2, upperBoundForB);

    return traveltime;
}

Traveltime* TraveltimeBuilder::buildOffsetContinuationTrajectory(
        const vector<float>& lowerBounds,
        const vector<float>& upperBounds,
        float h0
) {

    Traveltime* traveltime = new OffsetContinuationTrajectory();
    unsigned int modelParameterCount = traveltime->getNumberOfParameters();

    if (lowerBounds.size() != modelParameterCount || upperBounds.size() != modelParameterCount) {
        ostringstream exceptionString;
        exceptionString << "CMP requires exact " << modelParameterCount << " lower and upper bounds.";
        throw logic_error(exceptionString.str());
    }

    for (unsigned int i = 0; i < modelParameterCount; i++) {
        traveltime->updateLowerBoundForParameter(i, lowerBounds[i]);
        traveltime->updateUpperBoundForParameter(i, upperBounds[i]);
    }

    traveltime->updateReferenceHalfoffset(h0);

    return traveltime;
}
