#include "common/include/gpu/interface.h"
#include "cuda/include/semblance/kernel/cmp/differential_evolution.cuh"

__global__
void computeSemblancesForCommonMidPoint(
    const float *samples,
    const float *halfoffsetSquared,
    unsigned int startingTraceIndex,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    unsigned int individualsPerPopulation,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    unsigned int numberOfCommonResults,
    const float *x,
    float *fx
) {
    unsigned int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;

    if (sampleIndex < samplesPerTrace) {
        float semblance = 0;
        float stack = 0;

        unsigned int parameterIndex = sampleIndex * individualsPerPopulation + individualIndex;

        float t0 = static_cast<float>(sampleIndex) * dtInSeconds;
        float v = x[parameterIndex];
        float c = 4.0f / (v * v);

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        RESET_SEMBLANCE_NUM_COMP(numeratorComponents, MAX_WINDOW_SIZE);

        unsigned int usedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            unsigned int traceIndexWithOffset = startingTraceIndex + traceIndex;

            float h_sq = halfoffsetSquared[traceIndexWithOffset];
            const float *traceSamples = samples + traceIndexWithOffset * samplesPerTrace;

            float t = sqrt(t0 * t0 + c * h_sq);

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

        REDUCE_SEMBLANCE_STACK(numeratorComponents, linearSum, denominatorSum, windowSize, usedCount, semblance, stack);

        unsigned int offset = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfCommonResults;
        fx[offset] = semblance;
        fx[offset + 1] = stack;
    }
}

__global__
void selectBestIndividualsForCommonMidPoint(
    const float* x,
    const float* fx,
    float* resultArray,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfCommonResults
) {
    unsigned int sampleIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (sampleIndex < samplesPerTrace) {
        unsigned int popIndex = sampleIndex * individualsPerPopulation;
        unsigned int fitnessIndex = sampleIndex * individualsPerPopulation * numberOfCommonResults;

        float bestSemblance = -1, bestStack, bestVelocity;

        for (unsigned int individualIndex = 0; individualIndex < individualsPerPopulation; individualIndex++) {
            unsigned int featureOffset = fitnessIndex + individualIndex * numberOfCommonResults;

            float semblance = fx[featureOffset];
            float stack = fx[featureOffset + 1];
            float velocity = x[popIndex + individualIndex];

            if (semblance > bestSemblance) {
                bestSemblance = semblance;
                bestStack = stack;
                bestVelocity = velocity;
            }
        }

        resultArray[sampleIndex] = bestSemblance;
        resultArray[samplesPerTrace + sampleIndex] = bestStack;
        resultArray[2 * samplesPerTrace + sampleIndex] = bestVelocity;
    }
}
