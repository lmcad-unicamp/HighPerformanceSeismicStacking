#pragma once

#include "common/include/capability.h"
#include "common/include/output/Logger.hpp"
#include "common/include/model/Gather.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"
#include "common/include/semblance/algorithm/ComputeAlgorithmBuilder.hpp"
#include "common/include/semblance/data/DeviceContext.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <boost/program_options.hpp>
#include <memory>
#include <string>

using namespace std;

class Parser {
    protected:
        boost::program_options::options_description arguments;
        boost::program_options::variables_map argumentMap;

    public:
        Parser();
        virtual ~Parser();

        void parseArguments(int argc, const char *argv[]);

        unsigned int getNumberOfDevices() const;

        const string getFilename() const;

        const string getInputFilePath() const;

        const string getOutputDirectory() const;

        LogLevel getLogVerbosity() const;

        virtual Traveltime* parseTraveltime() const;

        void printHelp() const;

        void readGather() const;

        static enum traveltime_t traveltimeFromString(const string& model);

        virtual ComputeAlgorithm* parseComputeAlgorithm(
            ComputeAlgorithmBuilder* builder,
            shared_ptr<DeviceContext> deviceContext,
            shared_ptr<Traveltime> traveltime
        ) const = 0;

        virtual const string getParserType() const = 0;
};
