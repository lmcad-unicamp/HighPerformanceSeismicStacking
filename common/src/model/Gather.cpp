#include "common/include/model/Gather.hpp"
#include "common/include/model/Utils.hpp"
#include "common/include/output/Logger.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <math.h>
#include <string>
#include <sstream>

using namespace std;

unique_ptr<Gather> Gather::instance = nullptr;

Gather* Gather::getInstance() {
    if (instance == nullptr) {
        instance = make_unique<Gather>();
    }
    return instance.get();
}

Gather::Gather() :
    apm(0),
    aph(0),
    azimuthInDegrees(0),
    azimuthInRad(0),
    dt(0),
    dtInSeconds(0),
    lowestAllowedMidpoint(numeric_limits<float>::min()),
    biggestAllowedMidpoint(numeric_limits<float>::max()),
    samplesCount(0),
    samplesPerTrace(0),
    tau(0),
    isRead(false) {
}

void Gather::add(const Trace& trace) {
    traces.push_back(trace);
    samplesCount += static_cast<unsigned int>(trace.getSampleCount());
}

bool Gather::isGatherRead() const {
    return isRead;
}

void Gather::readGatherFromFile(const string& inFile) {
    ostringstream stringStream;

    ifstream fin;
    fin.open(inFile, ios::binary);

    if (!fin.is_open()) {
        throw invalid_argument(inFile + " couldn't be read.");
    }

    samplesCount = 0;

    while (!fin.eof()) {

        Trace trace;
        float m, h;

        trace.read(fin, azimuthInRad);
        m = trace.getMidpoint();
        h = trace.getHalfoffset();

        if (!samplesPerTrace) {
            samplesPerTrace = trace.getSampleCount();
            dt = trace.getInfo().dt;
            dtInSeconds = static_cast<float>(dt) * MICRO2SECONDS;

            tauIndexDisplacement = static_cast<int>(tau / dtInSeconds);
            windowSize = 2 * tauIndexDisplacement + 1;
        }

        map<float, Cdp>::iterator it = cdps.find(trace.getMidpoint());

        if (it == cdps.end()) {

            trace_info_t cdpInfo = trace.getInfo();

            cdpInfo.offset = 0;
            cdpInfo.sx = cdpInfo.gx = (trace.getInfo().sx + trace.getInfo().gx) >> 1;
            cdpInfo.sy = cdpInfo.gy = (trace.getInfo().sy + trace.getInfo().gy) >> 1;

            Cdp cdp(m, cdpInfo);

            LOGH("CDP not found. Creating a new one with midpoint = " << m);

            cdps[m] = cdp;
        }

        if (samplesPerTrace != trace.getSampleCount()) {
            // All traces must have the same amount of data events.
            continue;
        }

        if (abs(h) > aph) {
            LOGH("Discarding a trace of CDP (m = " << m << ") (|" << h << "| > " << aph << ")");
            continue;
        }

        if (!trace.isMidpointInRange(lowestAllowedMidpoint, biggestAllowedMidpoint)) {
            //LOGH("discarding a trace of CDP %d (out of limits)", trace.getInfo());
            continue;
        }

        add(trace);

        cdps[m].incrementTraceCountBy(1);
        cdps[m].incrementSampleCountBy(trace.getSampleCount());
    }

    std::sort(traces.begin(), traces.end());

    LOGI("Number of CDPs = " << cdps.size());
    LOGI("Number of traces in the gather = " << traces.size());
    LOGI("Number of samples per trace = " << samplesPerTrace);

    fin.close();

    isRead = true;
}

void Gather::setAzimuthInDegree(float azInDeg) {
    azimuthInDegrees = azInDeg;
    azimuthInRad = Utils::degreesToRad(azimuthInDegrees);
}

void Gather::setAzimuthInRad(float azInRad) {
    azimuthInRad = azInRad;
    azimuthInDegrees = Utils::radToDegrees(azimuthInRad);
}

const string Gather::toString() const {

    ostringstream stringStream;

    stringStream << "Samples per trace = " << samplesPerTrace << endl;
    stringStream << "Total traces count = " << getTotalTracesCount() << endl;
    stringStream << "Total samples = " << samplesCount << endl;
    stringStream << "CDPs = " << getTotalCdpsCount() << endl;

    stringStream << "apm = " << apm << " m" << endl;
    stringStream << "aph = " << aph << " m" << endl;
    stringStream << "tau = " << tau << endl;
    stringStream << "azimuth = " <<  azimuthInRad << " rad" << endl;

    if (lowestAllowedMidpoint > numeric_limits<float>::min()) {
        stringStream << "Smallest allowed midpoint = ";
        stringStream << lowestAllowedMidpoint << " m" << endl;
    }

    if (biggestAllowedMidpoint < numeric_limits<float>::max()) {
        stringStream << "Biggest allowed midpoint = ";
        stringStream << biggestAllowedMidpoint << " m" << endl;
    }

    return stringStream.str();
}

float Gather::getApm() const {
    return apm;
}

const map<float, Cdp>& Gather::getCdps() const {
    return cdps;
}

float Gather::getHalfoffsetOfTrace(unsigned int t) const {
    return traces[t].getHalfoffset();
}

float Gather::getMidpointOfTrace(unsigned int t) const {
    return traces[t].getMidpoint();
}

float Gather::getAzimuthInDegrees() const {
    return azimuthInDegrees;
}

float Gather::getAzimuthInRad() const {
    return azimuthInRad;
}

float Gather::getTau() const {
    return tau;
}

int Gather::getTauIndexDisplacement() const {
    return tauIndexDisplacement;
}

Trace& Gather::getTraceAtIndex(unsigned int i) {
    return traces[i];
}

unsigned int Gather::getTotalSamplesCount() const {
    return samplesCount;
}

unsigned int Gather::getTotalTracesCount() const {
    return static_cast<unsigned int>(traces.size());
}

unsigned int Gather::getTotalCdpsCount() const {
    return static_cast<unsigned int>(cdps.size());
}

unsigned int Gather::getSamplesPerTrace() const {
    return samplesPerTrace;
}

unsigned short Gather::getSamplePeriod() const {
    return dt;
}

float Gather::getSamplePeriodInSeconds() const {
    return dtInSeconds;
}

unsigned int Gather::getWindowSize() const {
    return windowSize;
}

float Gather::getLowestAllowedMidpoint() const {
    return lowestAllowedMidpoint;
}

float Gather::getBiggestAllowedMidpoint() const {
    return biggestAllowedMidpoint;
}

void Gather::setLowestAllowedMidpoint(float l) {
    lowestAllowedMidpoint = l;
}

void Gather::setBiggestAllowedMidpoint(float b) {
    biggestAllowedMidpoint = b;
}

void Gather::setApm(float apm) {
    this->apm = apm;
}

void Gather::setAph(float aph) {
    this->aph = aph;
}

void Gather::setTau(float tau) {
    this->tau = tau;
}
