#pragma once

#include "common/include/parser/Parser.hpp"
#include "common/include/semblance/data/DeviceContext.hpp"

#include <memory>

using namespace std;

class StretchFreeParser : public Parser {
    protected:
        static unique_ptr<Parser> instance;

    public:
        StretchFreeParser();

        Traveltime* parseTraveltime() const override;

        ComputeAlgorithm* parseComputeAlgorithm(
            ComputeAlgorithmBuilder* builder,
            shared_ptr<DeviceContext> deviceContext,
            shared_ptr<Traveltime> traveltime
        ) const override;

        const string getParserType() const override;

        static Parser* getInstance();
};
