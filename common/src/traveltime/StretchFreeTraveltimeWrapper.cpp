#include "common/include/traveltime/StretchFreeTraveltimeWrapper.hpp"

StretchFreeTraveltimeWrapper::StretchFreeTraveltimeWrapper(Traveltime* traveltime) : Traveltime({TraveltimeParameter("n")}) {
    wrappedTraveltime.reset(traveltime);
}

unsigned int StretchFreeTraveltimeWrapper::getNumberOfParameters() const {
    return wrappedTraveltime->getNumberOfParameters();
}

float StretchFreeTraveltimeWrapper::getReferenceHalfoffset() const {
    return wrappedTraveltime->getReferenceHalfoffset();
}

enum traveltime_t StretchFreeTraveltimeWrapper::getModel() const {
    return wrappedTraveltime->getModel();
}

const string StretchFreeTraveltimeWrapper::getTraveltimeWord() const {
    return "stretch_free_" + wrappedTraveltime->getTraveltimeWord();
}

void StretchFreeTraveltimeWrapper::updateReferenceHalfoffset(float h0) {
    wrappedTraveltime->updateReferenceHalfoffset(h0);
}

const string StretchFreeTraveltimeWrapper::toString() const {
    return wrappedTraveltime->toString();
}