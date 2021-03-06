#include "cuda/include/semblance/kernel/oct/common.cuh"
#include "cuda/include/semblance/kernel/oct/differential_evolution.cuh"

__global__
void computeSemblancesForOffsetContinuationTrajectory(
    const float *samples,
    const float *midpoint,
    const float *halfoffset,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    unsigned int individualsPerPopulation,
    float apm,
    float m0,
    float h0,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    unsigned int numberOfCommonResults,
    float* notUsedCountArray,
    const float *x,
    float *fx
) {
    unsigned int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;

    if (sampleIndex < samplesPerTrace) {
        enum gpu_error_code errorCode;

        float semblance = 0;
        float stack = 0;
        float mh, t;

        unsigned int parameterOffset = (sampleIndex * individualsPerPopulation + individualIndex) * 2;

        float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

        float v = x[parameterOffset];
        float c = 4.0f / (v * v);
        float slope = x[parameterOffset + 1];

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        RESET_SEMBLANCE_NUM_COMP(numeratorComponents, MAX_WINDOW_SIZE);

        unsigned int usedCount = 0, notUsedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            float m = midpoint[traceIndex];
            float h = halfoffset[traceIndex];
            const float *traceSamples = samples + traceIndex * samplesPerTrace;

            errorCode = computeDisplacedMidpoint(h, h0, t0, m0, c, slope, &mh);

            if (errorCode != NO_ERROR || fabs(m - mh) > apm) {
                notUsedCount++;
                continue;
            }

            errorCode = computeTime(h, h0, t0, m0, m, mh, c, slope, &t);

            if (errorCode == NO_ERROR) {
                COMPUTE_SEMBLANCE(
                    t,
                    dtInSeconds,
                    samplesPerTrace,
                    tauIndexDisplacement,
                    windowSize,
                    numeratorComponents,
                    linearSum,
                    denominatorSum,
                    usedCount
                );
            }
        }

        REDUCE_SEMBLANCE_STACK(numeratorComponents, linearSum, denominatorSum, windowSize, usedCount, semblance, stack);

        notUsedCountArray[threadIndex] += notUsedCount;

        unsigned int offset = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfCommonResults;
        fx[offset] = semblance;
        fx[offset + 1] = stack;
    }
}

__global__
void selectBestIndividualsForOffsetContinuationTrajectory(
    const float* x,
    const float* fx,
    float* resultArray,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfCommonResults
) {
    unsigned int sampleIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (sampleIndex < samplesPerTrace) {
        unsigned int popIndex = sampleIndex * individualsPerPopulation * 2;
        unsigned int fitnessIndex = sampleIndex * individualsPerPopulation * numberOfCommonResults;

        float bestSemblance = -1, bestStack, bestVelocity, bestSlope;

        for (unsigned int individualIndex = 0; individualIndex < individualsPerPopulation; individualIndex++) {
            unsigned int featureOffset = fitnessIndex + individualIndex * numberOfCommonResults;
            unsigned int individualOffset = popIndex + 2 * individualIndex;

            float semblance = fx[featureOffset];
            float stack = fx[featureOffset + 1];
            float velocity = x[individualOffset];
            float slope = x[individualOffset + 1];

            if (semblance > bestSemblance) {
                bestSemblance = semblance;
                bestStack = stack;
                bestVelocity = velocity;
                bestSlope = slope;
            }
        }

        resultArray[sampleIndex] = bestSemblance;
        resultArray[samplesPerTrace + sampleIndex] = bestStack;
        resultArray[2 * samplesPerTrace + sampleIndex] = bestVelocity;
        resultArray[3 * samplesPerTrace + sampleIndex] = bestSlope;
    }
}

__global__
void selectTracesForOffsetContinuationTrajectoryAndDifferentialEvolution(
    const float *midpointArray,
    const float *halfoffsetArray,
    const float *x,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    unsigned int individualsPerPopulation,
    float dtInSeconds,
    float apm,
    float m0,
    float h0,
    unsigned char* usedTraceMaskArray
) {
    unsigned int traceIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (traceIndex < traceCount) {

        float m, h, mh;
        m = midpointArray[traceIndex];
        h = halfoffsetArray[traceIndex];

        for (unsigned int sampleIndex = 0; !usedTraceMaskArray[traceIndex] && sampleIndex < samplesPerTrace; sampleIndex++) {
            float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

            for (unsigned int individualIndex = 0; !usedTraceMaskArray[traceIndex] && individualIndex < individualsPerPopulation; individualIndex++) {
                unsigned int individualOffset = (sampleIndex * individualsPerPopulation + individualIndex) * 2;

                float v = x[individualOffset];
                float c = 4.0f / (v * v);
                float slope = x[individualOffset + 1];

                enum gpu_error_code errorCode = computeDisplacedMidpoint(h, h0, t0, m0, c, slope, &mh);

                usedTraceMaskArray[traceIndex] = (errorCode == NO_ERROR && fabs(m - mh) <= apm);
            }
        }
    }
}
