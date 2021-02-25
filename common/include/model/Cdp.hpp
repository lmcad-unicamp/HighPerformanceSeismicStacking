#pragma once

#include "common/include/model/Trace.hpp"

class Cdp {
    private:
        float midpoint;

        trace_info_t cdpInfo;

        unsigned int sampleCount, traceCount;

    public:
        Cdp();

        Cdp(float midpnt, trace_info_t info);

        trace_info_t getCdpInfo() const;

        void incrementTraceCountBy(unsigned int s);
        void incrementSampleCountBy(unsigned int s);

        bool operator<(const Cdp& other) const;
        bool operator==(int other_id) const;
};
