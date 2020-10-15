#include "common/include/traveltime/Traveltime.hpp"

using namespace std;

unordered_map<SemblanceCommonResult, string> Traveltime::fixedResultDescription = {
    { SemblanceCommonResult::SEMBL, "coherence" },
    { SemblanceCommonResult::STACK, "stack" }
};

Traveltime::Traveltime(vector<TraveltimeParameter> v) : travelTimeParameters(v), referenceHalfOffset(0) {
}

unsigned int Traveltime::getIndexForCommonResult(SemblanceCommonResult r) const  {
    return static_cast<unsigned int>(r);
}

unsigned int Traveltime::getNumberOfParameters() const {
    return static_cast<unsigned int>(travelTimeParameters.size());
}

unsigned int Traveltime::getNumberOfCommonResults() const {
    return static_cast<unsigned int>(SemblanceCommonResult::CNT);
}

unsigned int Traveltime::getNumberOfResults() const {
    return getNumberOfCommonResults() + getNumberOfParameters();
}

const string Traveltime::getDescriptionForParameter(unsigned int i) const {
    return travelTimeParameters[i].getParameterDescription();
}

void Traveltime::updateLowerBoundForParameter(unsigned int p, float min) {
    travelTimeParameters[p].updateMinimum(min);
}

void Traveltime::updateUpperBoundForParameter(unsigned int p, float max) {
    travelTimeParameters[p].updateMaximum(max);
}

float Traveltime::getLowerBoundForParameter(unsigned int p) const {
    return travelTimeParameters[p].getMinimum();
}

float Traveltime::getUpperBoundForParameter(unsigned int p) const {
    return travelTimeParameters[p].getMaximum();
}

float Traveltime::getReferenceHalfoffset() const {
    return referenceHalfOffset;
}

const string Traveltime::getDescriptionForResult(unsigned int i) const {

    unsigned int fixedResultCount = static_cast<unsigned int>(SemblanceCommonResult::CNT);

    if (i < fixedResultCount) {
        return fixedResultDescription[static_cast<SemblanceCommonResult>(i)];
    }

    return travelTimeParameters[i - fixedResultCount].getParameterDescription();
}

gpu_traveltime_data_t Traveltime::toGpuData() const {
    return {
        .traveltime = getModel(),
        .numberOfParameters = getNumberOfParameters(),
        .numberOfCommonResults = getNumberOfCommonResults()
    };
}
