#pragma once

#include "common/include/model/Cdp.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define MICRO2SECONDS 1e-6f

using namespace std;

class Gather {
    private:
        float apm, aph;

        float azimuthInDegrees, azimuthInRad;

        unsigned short dt;
        float dtInSeconds;

        float lowestAllowedMidpoint, biggestAllowedMidpoint;

        unsigned int samplesCount, samplesPerTrace;

        float tau;

        int tauIndexDisplacement, windowSize;

        map<float, Cdp> cdps;

        vector<Trace> traces;

        static unique_ptr<Gather> instance;

        bool isRead;

    public:
        static Gather* getInstance();

        Gather();

        void add(const Trace& trace);

        float getApm() const;

        const map<float, Cdp>& getCdps() const;

        float getHalfoffsetOfTrace(unsigned int t) const;

        float getMidpointOfTrace(unsigned int t) const;

        float getAzimuthInDegrees() const;

        float getAzimuthInRad() const;

        float getTau() const;

        int getTauIndexDisplacement() const;

        Trace& getTraceAtIndex(unsigned int i);

        unsigned int getTotalSamplesCount() const;

        unsigned int getTotalTracesCount() const;

        unsigned int getTotalCdpsCount() const;

        unsigned int getSamplesPerTrace() const;

        unsigned short getSamplePeriod() const;

        float getSamplePeriodInSeconds() const;

        unsigned int getWindowSize() const;

        float getLowestAllowedMidpoint() const;

        float getBiggestAllowedMidpoint() const;

        bool isGatherRead() const;

        void readGatherFromFile(const string& inFile);

        void setApm(float apm);

        void setAph(float aph);

        void setTau(float tau);

        void setAzimuthInDegree(float azInDeg);

        void setAzimuthInRad(float azInRad);

        void setLowestAllowedMidpoint(float l);

        void setBiggestAllowedMidpoint(float b);

        const string toString() const;
};
