#include "common/include/model/Utils.hpp"
#include "common/include/traveltime/CommonReflectionSurface.hpp"

#include <sstream>

unsigned int CommonReflectionSurface::VELOCITY = 0;
unsigned int CommonReflectionSurface::A = 1;
unsigned int CommonReflectionSurface::B = 2;

CommonReflectionSurface::CommonReflectionSurface() :
    Traveltime({
        TraveltimeParameter("velocity"),
        TraveltimeParameter("a"),
        TraveltimeParameter("b")
    }) {
}

float CommonReflectionSurface::getAinInDegrees() const {
    return ainInDegrees;
}

float CommonReflectionSurface::getAinInRad() const {
    return ainInRadian;
}

float CommonReflectionSurface::getRatioForB() const {
    return bRatio;
}

float CommonReflectionSurface::getReferenceVelocity() const {
    return referenceVelocity;
}

void CommonReflectionSurface::setRatioForB(float p) {
    bRatio = p;
}

void CommonReflectionSurface::setReferenceVelocity(float v) {
    referenceVelocity = v;
}

void CommonReflectionSurface::setAinInDegrees(float ainInDeg) {
    ainInDegrees = ainInDeg;
    ainInRadian = Utils::degreesToRad(ainInDegrees);
}

void CommonReflectionSurface::setAinInRad(float ainInRad) {
    ainInRadian = ainInRad;
    ainInDegrees = Utils::radToDegrees(ainInRad);
}

enum traveltime_t CommonReflectionSurface::getModel() const {
    return ZOCRS;
}

const string CommonReflectionSurface::getTraveltimeWord() const {
    return "ZOCRS";
}

void CommonReflectionSurface::updateReferenceHalfoffset(float h0) {
}

const string CommonReflectionSurface::toString() const {

    ostringstream stringStream;

    stringStream << "Name = " << getTraveltimeWord() << endl;
    stringStream << "h0 = " << getReferenceHalfoffset() << endl;

    for (unsigned int i = 0; i < getNumberOfParameters(); i++) {
        stringStream << getDescriptionForParameter(i) << " in ";
        stringStream << "[" << getLowerBoundForParameter(i) << ", ";
        stringStream << getUpperBoundForParameter(i) << "]";
    }

    stringStream << "v0 = " << referenceVelocity << endl;
    stringStream << "ain = " << ainInRadian << " rad" << endl;
    stringStream << "bpctg = " << bRatio << endl;

    return stringStream.str();
}