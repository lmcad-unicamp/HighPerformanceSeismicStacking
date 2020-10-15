#pragma once

#include "common/include/traveltime/Traveltime.hpp"

using namespace std;

class OffsetContinuationTrajectory : public Traveltime {

    public:
        static unsigned int VELOCITY, SLOPE;

        OffsetContinuationTrajectory();

        enum traveltime_t getModel() const override;

        const string getTraveltimeWord() const override;

        void updateReferenceHalfoffset(float h0) override;

        const string toString() const override;
};
