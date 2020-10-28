#pragma once

#include "common/include/parser/Parser.hpp"

#include <memory>

using namespace std;

class DifferentialEvolutionParser : public Parser {
    protected:
        static unique_ptr<Parser> instance;

    public:
        DifferentialEvolutionParser();

        ComputeAlgorithm* parseComputeAlgorithm(
            ComputeAlgorithmBuilder* builder,
            shared_ptr<DeviceContext> deviceContext,
            shared_ptr<Traveltime> traveltime
        ) const override;

        const string getParserType() const override;

        static Parser* getInstance();
};
