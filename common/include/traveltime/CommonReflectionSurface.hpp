#pragma once

#include "common/include/traveltime/Traveltime.hpp"

using namespace std;

class CommonReflectionSurface : public Traveltime {

    private:
        float ainInDegrees, ainInRadian, referenceVelocity, bRatio;

    public:
        static unsigned int VELOCITY, A, B;

        CommonReflectionSurface();

        float getAinInDegrees() const;

        float getAinInRad() const;

        float getRatioForB() const;

        float getReferenceVelocity() const;

        void setAinInDegrees(float ainInDeg);

        void setAinInRad(float ainInRad);

        void setRatioForB(float p);

        void setReferenceVelocity(float v);

        enum traveltime_t getModel() const override;

        const string getTraveltimeWord() const override;

        void updateReferenceHalfoffset(float h0) override;

        const string toString() const override;
};
