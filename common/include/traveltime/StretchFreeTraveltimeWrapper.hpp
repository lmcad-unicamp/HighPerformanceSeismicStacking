#pragma once

#include "common/include/traveltime/Traveltime.hpp"
#include <memory>

using namespace std;

class StretchFreeTraveltimeWrapper : public Traveltime {

    private:
        unique_ptr<Traveltime> wrappedTraveltime;

    public:
        StretchFreeTraveltimeWrapper(Traveltime* traveltime);

        unsigned int getNumberOfParameters() const override;

        float getReferenceHalfoffset() const override;

        enum traveltime_t getModel() const override;

        const string getTraveltimeWord() const override;

        void updateReferenceHalfoffset(float h0) override;

        const string toString() const override;
};
