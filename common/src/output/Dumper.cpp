#include "common/include/output/Dumper.hpp"
#include "common/include/output/Logger.hpp"

#include <boost/filesystem.hpp>
#include <ctime>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace std;

Dumper::Dumper(const string& path, const string& dataFile, const string& computeMethod, const string& traveltime) {
    ostringstream outputFileStream;

    time_t t = time(NULL);
    tm* now = localtime(&t);

    outputFileStream << path << "/";
    outputFileStream << dataFile << "_";
    outputFileStream << computeMethod << "_";
    outputFileStream << traveltime << "_";
    outputFileStream << put_time(now, "%Y%m%d_%H%M");
    outputFileStream << "/";

    outputDirectoryPath = outputFileStream.str();
}

void Dumper::createDir() const {
    LOGI("Creating output directory at " << outputDirectoryPath);

    if (!boost::filesystem::create_directory(outputDirectoryPath)) {
        throw runtime_error("Directory couldn't be created");
    }
}

void Dumper::dumpAlgorithm(ComputeAlgorithm* algorithm) const {
    ofstream algorithmOutputFile;

    const string algorithmFile = outputDirectoryPath + "algorithm.txt";

    algorithmOutputFile.open(algorithmFile);

    if (!algorithmOutputFile.is_open()) {
        throw runtime_error("Couldn't open file " + algorithmFile + "to write algorithm parameters");
    }

    algorithmOutputFile << "==== Compute Algorithm Summary ====" << endl;
    algorithmOutputFile << algorithm->toString() << endl;

    algorithmOutputFile.close();
}

void Dumper::dumpGatherParameters(const string& file) const {
    Gather* gather = Gather::getInstance();

    const string gatherFile = outputDirectoryPath + "gather.txt";

    ofstream gatherOutputFile;
    gatherOutputFile.open(gatherFile);

    if (!gatherOutputFile.is_open()) {
        throw runtime_error("Couldn't open file " + gatherFile + " to write gather parameters");
    }

    gatherOutputFile << "==== Gather Summary ====" << endl;
    gatherOutputFile << "Input file = " << file << endl;
    gatherOutputFile << gather->toString() << endl;

    gatherOutputFile.close();
}

void Dumper::dumpResult(const string& resultName, const MidpointResult& result) const {
    Gather* gather = Gather::getInstance();

    const string resultFile = outputDirectoryPath + resultName + ".su";

    ofstream resultOutputFile;
    resultOutputFile.open(resultFile);

    if (!resultOutputFile.is_open()) {
        throw runtime_error("Couldn't open " + resultFile + " file to write result data.");
    }

    for (auto it = gather->getCdps().begin(); it != gather->getCdps().end(); it++) {

        float m0 = it->first;
        const Cdp& cdp = it->second;

        const vector<float>& samples = result.get(m0);

        trace_info_t cdpInfo = cdp.getCdpInfo();

        LOGD("Writing " << sizeof(trace_info_t) << " bytes related to header.");

        resultOutputFile.write(
            reinterpret_cast<const char*>(&cdpInfo),
            sizeof(trace_info_t)
        );

        LOGD("Writing " << samples.size() * sizeof(float) << " bytes related to data.");

        resultOutputFile.write(
            reinterpret_cast<const char*>(samples.data()),
            samples.size() * sizeof(float)
        );
    }

    resultOutputFile.close();
}

void Dumper::dumpStatisticalResult(const string& statResultName, const StatisticalMidpointResult& statResult) const {
    Gather* gather = Gather::getInstance();

    const string resultFile = outputDirectoryPath + statResultName + ".csv";

    ofstream resultOutputFile;
    resultOutputFile.open(resultFile);

    if (!resultOutputFile.is_open()) {
        throw runtime_error("Couldn't open " + resultFile + " file to write statistical result data.");
    }

    for (auto it = gather->getCdps().begin(); it != gather->getCdps().end(); it++) {
        float m0 = it->first;
        resultOutputFile << m0 << "," << statResult.getStatisticalResultForMidpoint(m0) << endl;
    }

    resultOutputFile.close();
}

void Dumper::dumpTraveltime(Traveltime* model) const {
    const string traveltimeFile = outputDirectoryPath + "traveltime.txt";

    ofstream modelOutputFile;
    modelOutputFile.open(traveltimeFile);

    if (!modelOutputFile.is_open()) {
        throw runtime_error("Couldn't open " + traveltimeFile + " to write traveltime model parameters.");
    }

    modelOutputFile << "==== Model Summary ====" << endl;
    modelOutputFile << model->toString() << endl;

    modelOutputFile.close();
}

const string& Dumper::getOutputDirectoryPath() const {
    return outputDirectoryPath;
}

// void Dumper::dumpExArguments(double totalElapsedTime) const {

//     ofstream argumentsOutputFile;
//     argumentsOutputFile.open(outputDirectoryPath + "/parameters.txt");

//     Arguments& const arguments = Arguments::getInstance();

//     if (argumentsOutputFile.is_open()) {

//         argumentsOutputFile << "** " << MethodTypeStrings[INDEX_OF(computeTechnology)];
//         argumentsOutputFile << " (" << MethodStrategyDescStrings[INDEX_OF(computeStrategy)] << ") **";
//         argumentsOutputFile << endl << endl;

//         argumentsOutputFile << "** Method " << MethodStrings[INDEX_OF(arguments.chosen_method)] << " **" << endl;
//         argumentsOutputFile << "** Common Parameters **" << endl;
//         argumentsOutputFile << "input data = " << arguments.input_file << endl;
//         argumentsOutputFile << "aph = " << arguments.aph << endl;
//         argumentsOutputFile << "tau = " << arguments.tau << endl;
//         argumentsOutputFile << "azimuth = " <<  arguments.azimuthInRadians << " rad" << endl;

//         if (computeStrategy != MethodStrategy::STACK) {

//             switch(arguments.chosen_method) {
//                 case Method::CMP:
//                     argumentsOutputFile << "vmin = " <<  arguments.vmin << endl;
//                     argumentsOutputFile << "vmax = " <<  arguments.vmax << endl;

//                     if (computeStrategy == MethodStrategy::EXHAUSTIVE ||
//                         computeStrategy == MethodStrategy::EXHAUSTIVE_GREEDY) {
//                         argumentsOutputFile << "nv = " <<  arguments.nv << endl;
//                     }
//                     break;

//                 case Method::CRS:
//                     argumentsOutputFile << endl << "** CRS parameters **" << endl << endl;
//                     argumentsOutputFile << "apm = " << arguments.apm << endl;
//                     argumentsOutputFile << "vmin = " << arguments.vmin << endl;
//                     argumentsOutputFile << "vmax = " << arguments.vmax << endl;
//                     argumentsOutputFile << "v0 = " << arguments.v0 << endl;
//                     argumentsOutputFile << "ain = " << arguments.ainInRadians << " rad" << endl;
//                     argumentsOutputFile << "bpctg = " << arguments.bpctg << endl;

//                     if (computeStrategy == MethodStrategy::EXHAUSTIVE ||
//                         computeStrategy == MethodStrategy::EXHAUSTIVE_GREEDY) {
//                         argumentsOutputFile << "nv = " << arguments.nv << endl;
//                         argumentsOutputFile << "na = " << arguments.na << endl;
//                         argumentsOutputFile << "nb = " << arguments.nb << endl;
//                     }
//                     break;

//                 case Method::CRP:
//                     argumentsOutputFile << endl << "** CRP parameters **" << endl << endl;
//                     argumentsOutputFile << "apm = " << arguments.apm << endl;
//                     argumentsOutputFile << "aph0 = " << arguments.aph0 << endl;
//                     argumentsOutputFile << "vmin = " << arguments.vmin << endl;
//                     argumentsOutputFile << "vmax = " << arguments.vmax << endl;
//                     argumentsOutputFile << "amin = " << arguments.amin << endl;
//                     argumentsOutputFile << "amax = " << arguments.amax << endl;
//                     argumentsOutputFile << "h0 = " << arguments.h0 << endl;

//                     if (computeStrategy == MethodStrategy::EXHAUSTIVE ||
//                         computeStrategy == MethodStrategy::EXHAUSTIVE_GREEDY) {
//                         argumentsOutputFile << "nv = " << arguments.nv << endl;
//                         argumentsOutputFile << "na = " << arguments.na << endl;
//                     }
//                     break;
//             }
//         }
//         else {
//             argumentsOutputFile << "nn = " << arguments.nn << endl;
//             argumentsOutputFile << "v_array = " << arguments.v_array << endl;

//             switch(arguments.chosen_method) {
//                 case Method::CRS:
//                     argumentsOutputFile << "b_array = " << arguments.b_array << endl;

//                 case Method::CRP:
//                     argumentsOutputFile << "a_array = " << arguments.a_array << endl;
//                     argumentsOutputFile << "apm = " << arguments.apm << endl;
//                     break;
//             }
//         }

//         if (arguments.cdp_min > 0) {
//             argumentsOutputFile << "cdpmin = " << arguments.cdp_min << endl;
//         }

//         if (arguments.cdp_max > 0) {
//             argumentsOutputFile << "cdpmax = " << arguments.cdp_max << endl;
//         }

//         if (computeStrategy == MethodStrategy::GA) {
//             argumentsOutputFile << endl << "** GA parameters **" << endl << endl;
//             argumentsOutputFile << "Generations = " << arguments.gen << endl;
//             argumentsOutputFile << "Individuals per population = " << arguments.pop_size << endl;
//         }

//         argumentsOutputFile << "duration = " << totalElapsedTime << " seconds" << endl;

//         argumentsOutputFile.close();
//     }
// }
