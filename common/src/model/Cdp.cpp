#include "common/include/model/Cdp.hpp"

Cdp::Cdp() : sampleCount(0), traceCount(0) {
}

Cdp::Cdp(float midpnt, trace_info_t info) :
    midpoint(midpnt),
    cdpInfo(info),
    sampleCount(0),
    traceCount(0) {
}

trace_info_t Cdp::getCdpInfo() const {
    return cdpInfo;
}

void Cdp::incrementTraceCountBy(unsigned int s) {
    traceCount += s;
}

void Cdp::incrementSampleCountBy(unsigned int s) {
    sampleCount += s;
}

bool Cdp::operator<(const Cdp& other) const {
    return (midpoint < other.midpoint);
}

bool Cdp::operator==(int other_id) const {
    return (cdpInfo.cdp == other_id);
}
