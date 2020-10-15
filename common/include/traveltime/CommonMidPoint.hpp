#pragma once

#include "common/include/traveltime/Traveltime.hpp"

using namespace std;

class CommonMidPoint : public Traveltime {

    public:
        static unsigned int VELOCITY;

        CommonMidPoint();

        enum traveltime_t getModel() const override;

        const string getTraveltimeWord() const override;

        void updateReferenceHalfoffset(float h0) override;

        const string toString() const override;
};
