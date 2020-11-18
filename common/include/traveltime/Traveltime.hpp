#pragma once

#include "common/include/capability.h"
#include "common/include/gpu/interface.h"
#include "common/include/traveltime/TraveltimeParameter.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

enum class SemblanceCommonResult { SEMBL, STACK, CNT };

class Traveltime {

    protected:
        static unordered_map<SemblanceCommonResult, string> fixedResultDescription;
        static unordered_map<SemblanceCommonResult, string> commonResultDescription;

        vector<TraveltimeParameter> travelTimeParameters;

        float referenceHalfOffset;

        Traveltime(vector<TraveltimeParameter> v);

    public:
        virtual ~Traveltime() {};

        unsigned int getIndexForCommonResult(SemblanceCommonResult r) const;

        virtual unsigned int getNumberOfParameters() const;

        unsigned int getNumberOfCommonResults() const;

        unsigned int getNumberOfResults() const;

        const string getDescriptionForParameter(unsigned int i) const;

        const string getDescriptionForResult(unsigned int i) const;

        virtual float getReferenceHalfoffset() const;

        void updateLowerBoundForParameter(unsigned int p, float min);

        void updateUpperBoundForParameter(unsigned int p, float max);

        float getLowerBoundForParameter(unsigned int p) const;

        float getUpperBoundForParameter(unsigned int p) const;

        // Virtual methods.

        virtual const string getTraveltimeWord() const = 0;

        virtual void updateReferenceHalfoffset(float h0) = 0;

        virtual enum traveltime_t getModel() const = 0;

        virtual const string toString() const = 0;
};
