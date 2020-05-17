#include "common/include/traveltime/CommonMidPoint.hpp"
#include <sstream>

CommonMidPoint::CommonMidPoint() : Traveltime({TraveltimeParameter("velocity")}) {
}

enum traveltime_t CommonMidPoint::getModel() const {
    return CMP;
}

const string CommonMidPoint::getTraveltimeWord() const {
    return "CMP";
}

void CommonMidPoint::updateReferenceHalfoffset(float h0) {
}

const string CommonMidPoint::toString() const {

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
