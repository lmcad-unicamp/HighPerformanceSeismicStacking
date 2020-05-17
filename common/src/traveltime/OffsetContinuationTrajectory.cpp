#include "common/include/traveltime/OffsetContinuationTrajectory.hpp"

#include <sstream>

OffsetContinuationTrajectory::OffsetContinuationTrajectory() :
    Traveltime({
        TraveltimeParameter("velocity"),
        TraveltimeParameter("slope")
    }) {
}

enum traveltime_t OffsetContinuationTrajectory::getModel() const {
    return OCT;
}

const string OffsetContinuationTrajectory::getTraveltimeWord() const {
    return "OCT";
}

void OffsetContinuationTrajectory::updateReferenceHalfoffset(float h0) {
    referenceHalfOffset = h0;
};

const string OffsetContinuationTrajectory::toString() const {

    ostringstream stringStream;

    stringStream << "Name = " << getTraveltimeWord() << endl;
    stringStream << "h0 = " << getReferenceHalfoffset() << endl;

    for (unsigned int i = 0; i < getNumberOfParameters(); i++) {
        stringStream << getDescriptionForParameter(i) << " in ";
        stringStream << "[" << getLowerBoundForParameter(i) << ", ";
        stringStream << getUpperBoundForParameter(i) << "]";
    }

    return stringStream.str();

}