#include "common/include/gpu/interface.h"

__kernel
void computeSemblancesForCommonMidPoint(
    __global float *samples,
    __global float *halfoffsetSquared,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    unsigned int individualsPerPopulation,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    unsigned int numberOfCommonResults,
    __global float *x,
    __global float *fx
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;

    if (sampleIndex < samplesPerTrace) {
        float semblance = 0;
        float stack = 0;

        unsigned int parameterIndex = sampleIndex * individualsPerPopulation + individualIndex;

        float t0 = ((float) sampleIndex) * dtInSeconds;
        float v = x[parameterIndex];
        float c = 4.0f / (v * v);

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        RESET_SEMBLANCE_NUM_COMP(numeratorComponents, MAX_WINDOW_SIZE);

        unsigned int usedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            float h_sq = halfoffsetSquared[traceIndex];
            __global float *traceSamples = samples + traceIndex * samplesPerTrace;

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

__kernel
void selectBestIndividualsForCommonMidPoint(
    __global float* x,
    __global float* fx,
    __global float* resultArray,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfCommonResults
) {
    unsigned int sampleIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

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
