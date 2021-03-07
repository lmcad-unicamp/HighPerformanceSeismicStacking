#include "common/include/parser/Parser.hpp"
#include "common/include/model/Utils.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/traveltime/TraveltimeBuilder.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>

using namespace boost::program_options;
using namespace std;

Parser::Parser() : arguments("Allowed options") {
    arguments.add_options()
        ("aph", value<float>()->required(), "APH constant.")
        ("apm", value<float>()->required(), "APM constant.")
        ("ain", value<float>(), "Angle in degrees to be used during computation of A for ZO-CRS.")
        ("azimuth", value<float>()->required(), "Azimuth angle in degrees.")
        ("bpctg", value<float>(), "Percentage of C that will be used to compute B for ZO-CRS.")
        ("h0", value<float>(), "h0 constant.")
        ("help", "Produce help message.")
        ("highest-midpoint", value<float>(), "Higher bound value for a CDP midpoint.")
        ("input", value<string>()->required(), "*.su input data file.")
        ("lower-bounds", value<vector<float>>()->required()->multitoken(), "Lower bound for traveltime parameters.")
        ("lowest-midpoint", value<float>(), "Lower bound value for a CDP midpoint.")
        ("kernel-path", value<string>(), "OpenCL's kernel location.")
        ("output", value<string>()->required(), "*.su output data prefix.")
        ("tau", value<float>(), "Semblance processing window given by w = 2 * tau + 1.")
        ("thread-count", value<unsigned int>()->default_value(DEFAULT_THREAD_COUNT), "GPU thread count.")
        ("traveltime", value<string>()->required(), "Traveltime traveltime to be used.")
        ("upper-bounds", value<vector<float>>()->required()->multitoken(), "Upper bounds for traveltime parameters.")
        ("v0", value<float>(), "Velocity used to compute ZO-CRS A parameter.")
        ("verbose", value<unsigned int>(), "Verbosity level. Must be in the [0..4] integer range.")
    ;
}

Parser::~Parser() {
}

void Parser::parseArguments(int argc, const char *argv[]) {
    try {
        int denyShortStyle = command_line_style::unix_style ^ command_line_style::allow_short;
        store(parse_command_line(argc, argv, arguments, denyShortStyle), argumentMap);
        notify(argumentMap);
        Logger::getInstance()->setVerbosity(getLogVerbosity());
    } catch (const exception& ex) {
        ostringstream exceptionString;
        exceptionString << "Unable to parse input arguments: " << ex.what();
        throw invalid_argument(exceptionString.str());
    }
}

const string Parser::getFilename() const {
    return filesystem::path(getInputFilePath()).stem();
}

unsigned int Parser::getGpuThreadCount() const {
    return argumentMap["thread-count"].as<unsigned int>();
}

const string Parser::getInputFilePath() const {
    if (argumentMap.count("input")) {
        return argumentMap["input"].as<string>();
    }

    throw invalid_argument("Input data file not provided.");
}

const string Parser::getOutputDirectory() const {
    if (argumentMap.count("output")) {
        return argumentMap["output"].as<string>();
    }

    throw invalid_argument("Output path not provided.");
}

LogLevel Parser::getLogVerbosity() const {
    if (argumentMap.count("verbose")) {
        unsigned int verboseLevel = min(argumentMap["verbose"].as<unsigned int>(), static_cast<unsigned int>(LogLevel::DEBUG));
        return static_cast<LogLevel>(verboseLevel);
    }

    return LogLevel::INFO;
}

Traveltime* Parser::parseTraveltime() const {
    if (argumentMap.count("traveltime")) {

        TraveltimeBuilder* builder = TraveltimeBuilder::getInstance();

        enum traveltime_t traveltime = traveltimeFromString(argumentMap["traveltime"].as<string>());

        const vector<float>& lowerBounds = argumentMap["lower-bounds"].as<vector<float>>();
        const vector<float>& upperBounds = argumentMap["upper-bounds"].as<vector<float>>();

        if (traveltime == CMP) {
            return builder->buildCommonMidPoint(lowerBounds, upperBounds);
        }
        else if (traveltime == ZOCRS) {

            if (!argumentMap.count("ain") || !argumentMap.count("v0") || !argumentMap.count("bpctg")) {
                throw invalid_argument("Not all arguments have been provided.");
            }

            float v0 = argumentMap["v0"].as<float>();
            float aIn = argumentMap["ain"].as<float>();
            float bPctg = argumentMap["bpctg"].as<float>();

            return builder->buildZeroOffsetCommonReflectionSurface(lowerBounds, upperBounds, v0, aIn, bPctg);
        }
        else if (traveltime == OCT) {

            float h0 = 0;

            if (argumentMap.count("h0")) {
                h0 = argumentMap["h0"].as<float>();
            }

            return builder->buildOffsetContinuationTrajectory(lowerBounds, upperBounds, h0);
        }
    }

    throw invalid_argument("Traveltime traveltime not provided.");
}

void Parser::printHelp() const {
    cout << arguments << endl;
}

void Parser::readGather() const {
    if (!argumentMap.count("apm") || !argumentMap.count("aph") || !argumentMap.count("tau")) {
        throw invalid_argument("Not all arguments to read the gather have been provided.");
    }

    Gather* gather = Gather::getInstance();

    gather->setApm(argumentMap["apm"].as<float>());
    gather->setAph(argumentMap["aph"].as<float>());
    gather->setTau(argumentMap["tau"].as<float>());

    if (argumentMap.count("azimuth")) {
        gather->setAzimuthInDegree(argumentMap["azimuth"].as<float>());
    }

    if (argumentMap.count("lowest-midpoint")) {
        gather->setLowestAllowedMidpoint(argumentMap["lowest-midpoint"].as<float>());
    }

    if (argumentMap.count("highest-midpoint")) {
        gather->setBiggestAllowedMidpoint(argumentMap["highest-midpoint"].as<float>());
    }

    gather->readGatherFromFile(getInputFilePath());
}

enum traveltime_t Parser::traveltimeFromString(const string& traveltime) {
    ostringstream exceptionString;

    if (traveltime == "cmp") {
        return CMP;
    }
    else if (traveltime == "zocrs") {
        return ZOCRS;
    }
    else if (traveltime == "oct") {
        return OCT;
    }

    exceptionString << "Invalid traveltime: " << traveltime;

    throw invalid_argument(exceptionString.str());
}
