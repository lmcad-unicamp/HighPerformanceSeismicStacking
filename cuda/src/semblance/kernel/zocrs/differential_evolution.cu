#include "common/include/gpu/interface.h"
#include "cuda/include/semblance/kernel/zocrs/differential_evolution.cuh"

__global__
void computeSemblancesForZeroOffsetCommonReflectionSurface(
    const float *samples,
    const float *midpoint,
    const float *halfoffsetSquared,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    unsigned int individualsPerPopulation,
    float m0,
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

        unsigned int parameterOffset = (sampleIndex * individualsPerPopulation + individualIndex) * 3;

        float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

        float v = x[parameterOffset];
        float c = 4.0f / (v * v);
        float a = x[parameterOffset + 1];
        float b = x[parameterOffset + 2];

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        RESET_SEMBLANCE_NUM_COMP(numeratorComponents, MAX_WINDOW_SIZE);

        unsigned int usedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            float m = midpoint[traceIndex];
            float h_sq = halfoffsetSquared[traceIndex];
            const float *traceSamples = samples + traceIndex * samplesPerTrace;

            float dm = m - m0;
            float tmp = t0 + a * dm;
            tmp = tmp * tmp + b * dm * dm + c * h_sq;

            if (tmp >= 0) {
                float t = sqrt(tmp);
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

        unsigned int offset = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfCommonResults;
        fx[offset] = semblance;
        fx[offset + 1] = stack;
    }
}

__global__
void selectBestIndividualsForZeroOffsetCommonReflectionSurface(
    const float* x,
    const float* fx,
    float* resultArray,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfCommonResults
) {
    unsigned int sampleIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (sampleIndex < samplesPerTrace) {
        unsigned int popIndex = sampleIndex * individualsPerPopulation * 3;
        unsigned int fitnessIndex = sampleIndex * individualsPerPopulation * numberOfCommonResults;

        float bestSemblance = -1, bestStack, bestVelocity, bestA, bestB;

        for (unsigned int individualIndex = 0; individualIndex < individualsPerPopulation; individualIndex++) {
            unsigned int featureOffset = fitnessIndex + individualIndex * numberOfCommonResults;
            unsigned int individualOffset = popIndex + 3 * individualIndex;

            float semblance = fx[featureOffset];
            float stack = fx[featureOffset + 1];
            float velocity = x[individualOffset];
            float a = x[individualOffset + 1];
            float b = x[individualOffset + 2];

            if (semblance > bestSemblance) {
                bestSemblance = semblance;
                bestStack = stack;
                bestVelocity = velocity;
                bestA = a;
                bestB = b;
            }
        }

        resultArray[sampleIndex] = bestSemblance;
        resultArray[samplesPerTrace + sampleIndex] = bestStack;
        resultArray[2 * samplesPerTrace + sampleIndex] = bestVelocity;
        resultArray[3 * samplesPerTrace + sampleIndex] = bestA;
        resultArray[4 * samplesPerTrace + sampleIndex] = bestB;
    }
}
