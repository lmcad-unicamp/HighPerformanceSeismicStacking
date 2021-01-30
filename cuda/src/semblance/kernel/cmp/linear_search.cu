#include "common/include/gpu/interface.h"
#include "cuda/include/semblance/kernel/cmp/linear_search.cuh"

__global__
void buildParameterArrayForCommonMidPoint(
    float* parameterArray,
    float minVelocity,
    float increment,
    unsigned int totalParameterCount
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIndex < totalParameterCount) {
        float v = minVelocity + static_cast<float>(threadIndex) * increment;
        parameterArray[threadIndex] = 4.0f / (v * v);
    }
}

__global__
void computeSemblancesForCommonMidPoint(
    const float *samples,
    const float *halfoffsetSquared,
    unsigned int startingTraceIndex,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    /* Parameter arrays */
    const float *parameterArray,
    unsigned int totalParameterCount,
    /* Output arrays */
    float *semblanceArray,
    float *stackArray
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int sampleIndex = threadIndex / totalParameterCount;
    unsigned int parameterIndex = threadIndex % totalParameterCount;

    if (sampleIndex < samplesPerTrace) {
        float semblance = 0;
        float stack = 0;

        float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

        float c = parameterArray[parameterIndex];

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

        unsigned int offset = sampleIndex * totalParameterCount + parameterIndex;
        semblanceArray[offset] = semblance;
        stackArray[offset] = stack;
    }
}

__global__
void selectBestSemblancesForCommonMidPoint(
    const float *semblanceArray,
    const float *stackArray,
    const float *parameterArray,
    unsigned int totalParameterCount,
    unsigned int samplesPerTrace,
    float *resultArray
) {
    unsigned int sampleIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (sampleIndex < samplesPerTrace) {
        unsigned int offset = sampleIndex * totalParameterCount;

        float bestSemblance = -1, bestStack, bestVelocity;

        for (unsigned int parameterIndex = 0; parameterIndex < totalParameterCount; parameterIndex++) {
            float semblance = semblanceArray[offset + parameterIndex];
            float stack = stackArray[offset + parameterIndex];
            float c = parameterArray[parameterIndex];

            if (semblance > bestSemblance) {
                bestSemblance = semblance;
                bestStack = stack;
                bestVelocity = 2 / sqrt(c);
            }
        }

        resultArray[sampleIndex] = bestSemblance;
        resultArray[samplesPerTrace + sampleIndex] = bestStack;
        resultArray[2 * samplesPerTrace + sampleIndex] = bestVelocity;
    }
}
