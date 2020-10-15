#pragma once

#include "common/include/model/Gather.hpp"
#include "common/include/semblance/data/DeviceContext.hpp"
#include "common/include/semblance/data/DataContainerBuilder.hpp"
#include "common/include/semblance/result/StatisticResult.hpp"
#include "common/include/traveltime/Traveltime.hpp"

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>

using namespace std;

enum class GatherData {
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

        unsigned int threadCount, threadCountToRestore;

        //
        // Data for a single m0.
        //

        unique_ptr<DataContainer> deviceParameterArray, deviceResultArray, deviceNotUsedCountArray;

        unordered_map<StatisticResult, float> computedStatisticalResults;

        unordered_map<GatherData, unique_ptr<DataContainer>> deviceFilteredTracesDataMap;

        unsigned int filteredTracesCount;

        vector<float> computedResults;

        //
        // Functions
        //

        void changeThreadCountTemporarilyTo(unsigned int t);

        void restoreThreadCount();

    public:
        ComputeAlgorithm(
            const string& name,
            shared_ptr<Traveltime> model,
            shared_ptr<DeviceContext> context,
            DataContainerBuilder* dataBuilder
        );

        virtual ~ComputeAlgorithm();

        bool isSetUp() const;

        void copyGatherDataToDevice();

        void copyOnlySelectedTracesToDevice(const vector<unsigned char>& usedTraceMask);

        const string& getAlgorithmName() const { return algorithmName; };

        const vector<float>& getComputedResults() const { return computedResults; };

        float getStatisticalResult(StatisticResult statResult) const;

        void saveStatisticalResults(
            unsigned long totalUsedTracesCount,
            chrono::duration<double> totalExecutionTime,
            chrono::duration<double> selectionExecutionTime
        );

        void setDeviceSourcePath(const string& path);

        //
        // Virtual methods.
        //

        virtual void computeSemblanceAndParametersForMidpoint(float m0) = 0;
        virtual void computeSemblanceAtGpuForMidpoint(float m0) = 0;
        virtual unsigned int getParameterArrayStep() const = 0;
        virtual void selectTracesToBeUsedForMidpoint(float m0) = 0;
        virtual void setUp() = 0;
        virtual const string toString() const = 0;
};
