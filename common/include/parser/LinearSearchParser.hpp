#pragma once

#include "common/include/parser/Parser.hpp"

using namespace std;

class LinearSearchParser : public Parser {
    protected:
        static unique_ptr<Parser> instance;

    public:
        LinearSearchParser();

        ComputeAlgorithm* parseComputeAlgorithm(
            ComputeAlgorithmBuilder* builder,
            shared_ptr<DeviceContext> deviceContext,
            shared_ptr<Traveltime> traveltime
        ) const override;

        const string getParserType() const override;

        static Parser* getInstance();
};
