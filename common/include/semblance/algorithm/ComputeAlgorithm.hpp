#pragma once

#include "common/include/model/Gather.hpp"
#include "common/include/semblance/data/DeviceContext.hpp"
#include "common/include/semblance/data/DataContainerBuilder.hpp"
#include "common/include/semblance/result/StatisticResult.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>

using namespace std;

#define DEFAULT_THREAD_COUNT 64

enum class GatherData {
    SAMPL,
    MDPNT,
    HLFOFFST,
    HLFOFFST_SQ,
    FILT_SAMPL,
    FILT_MDPNT,
    FILT_HLFOFFST,
    FILT_HLFOFFST_SQ,
    CNT
};

class ComputeAlgorithm {
    protected:
        bool isSet;

        DataContainerBuilder* dataFactory;

        string algorithmName, deviceSource;

        shared_ptr<DeviceContext> deviceContext;

        shared_ptr<Traveltime> traveltime;

        unsigned int threadCount;

        //
        // Data for a single m0.
        //

        unique_ptr<DataContainer> deviceParameterArray, deviceResultArray, deviceNotUsedCountArray;

        unordered_map<StatisticResult, float> computedStatisticalResults;

        unordered_map<GatherData, unique_ptr<DataContainer>> deviceFilteredTracesDataMap;

        unsigned int filteredTracesCount, startingTraceIndex;

        vector<float> computedResults;

        //
        // Functions
        //

        pair<unsigned int, unsigned int> selectTracesContinuous(float m0) const;

    public:
        ComputeAlgorithm(
            const string& name,
            shared_ptr<Traveltime> model,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder,
            unsigned int threadCount = DEFAULT_THREAD_COUNT
        );

        virtual ~ComputeAlgorithm();

        bool isSetUp() const;

        void copyGatherDataToDevice();

        void copyOnlySelectedTracesToDevice(const vector<unsigned char>& usedTraceMask);

        const string& getAlgorithmName() const { return algorithmName; };

        const vector<float>& getComputedResults() const { return computedResults; };

        float getStatisticalResult(StatisticResult statResult) const;

        void saveStatisticalResults(
            float totalUsedTracesCount,
            chrono::duration<double> totalExecutionTime,
            chrono::duration<double> selectionExecutionTime
        );

        //
        // Virtual methods.
        //
        virtual void compileKernels(const string& deviceKernelSourcePath);

        virtual void computeSemblanceAndParametersForMidpoint(float m0) = 0;
        virtual void computeSemblanceAtGpuForMidpoint(float m0) = 0;
        virtual void selectTracesToBeUsedForMidpoint(float m0) = 0;
        virtual void setUp() = 0;
        virtual const string toString() const = 0;
};
