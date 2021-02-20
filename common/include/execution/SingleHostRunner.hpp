#pragma once

#include "common/include/parser/Parser.hpp"
#include "common/include/model/Gather.hpp"
#include "common/include/semblance/data/DeviceContextBuilder.hpp"
#include "common/include/semblance/result/ResultSet.hpp"

#include <memory>
#include <mutex>
#include <queue>

using namespace std;

class SingleHostRunner {
    protected:
        Parser* parser;

        mutex resultSetMutex, deviceMutex;

        vector<queue<float>> midpointQueues;

        ComputeAlgorithmBuilder* algorithmBuilder;

        DeviceContextBuilder* deviceContextBuilder;

        shared_ptr<Traveltime> traveltime;

        unique_ptr<ResultSet> resultSet;

        unsigned int deviceIndex;

    public:
        SingleHostRunner(
            Parser* parser,
            ComputeAlgorithmBuilder* algorithmBuilder,
            DeviceContextBuilder* deviceContextBuilder
        );

        int main(int argc, const char *argv[]);

        ComputeAlgorithm* getComputeAlgorithm();

        ResultSet* getResultSet();

        queue<float>& getMidpointQueue(unsigned int threadId);

        mutex& getDeviceMutex();

        mutex& getQueueMutex();

        mutex& getResultSetMutex();

        static void workerThread(SingleHostRunner *ref, unsigned int threadIndex);

        //
        // Virtual methods.
        //

        virtual unsigned int getNumOfDevices() const = 0;
};
