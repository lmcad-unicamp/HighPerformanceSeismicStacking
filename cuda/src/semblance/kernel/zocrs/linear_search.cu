#include "common/include/gpu/interface.h"
#include "cuda/include/semblance/kernel/zocrs/linear_search.cuh"

__global__
void buildParameterArrayForZeroOffsetCommonReflectionSurface(
    float* parameterArray,
    float minVelocity,
    float incrementVelocity,
    unsigned int countVelocity,
    float minA,
    float incrementA,
    unsigned int countA,
    float minB,
    float incrementB,
    unsigned int countB,
    unsigned int totalParameterCount
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIndex < totalParameterCount) {
        unsigned int idxVelocity = (threadIndex / (countA * countB)) % countVelocity;
        unsigned int idxA = (threadIndex / countB) % countA;
        unsigned int idxB = threadIndex % countB;

        float v = minVelocity + static_cast<float>(idxVelocity) * incrementVelocity;
        float a = minA + static_cast<float>(idxA) * incrementA;
        float b = minB + static_cast<float>(idxB) * incrementB;

        unsigned int offset = 3 * threadIndex;
        parameterArray[offset] = 4.0f / (v * v);
        parameterArray[offset + 1] = a;
        parameterArray[offset + 2] = b;
    }
}

__global__
void computeSemblancesForZeroOffsetCommonReflectionSurface(
    const float *samples,
    const float *midpoint,
    const float *halfoffsetSquared,
    unsigned int startingTraceIndex,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float m0,
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
    unsigned int parameterOffset = 3 * parameterIndex;

    if (sampleIndex < samplesPerTrace) {
        float semblance = 0;
        float stack = 0;

        float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

        float c = parameterArray[parameterOffset];
        float a = parameterArray[parameterOffset + 1];
        float b = parameterArray[parameterOffset + 2];

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        RESET_SEMBLANCE_NUM_COMP(numeratorComponents, MAX_WINDOW_SIZE);

        unsigned int usedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            unsigned int traceIndexWithOffset = startingTraceIndex + traceIndex;

            float m = midpoint[traceIndexWithOffset];
            float h_sq = halfoffsetSquared[traceIndexWithOffset];
            const float *traceSamples = samples + traceIndexWithOffset * samplesPerTrace;

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

        unsigned int offset = sampleIndex * totalParameterCount + parameterIndex;
        semblanceArray[offset] = semblance;
        stackArray[offset] = stack;
    }
}

__global__
void selectBestSemblancesForZeroOffsetCommonReflectionSurface(
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

        float bestSemblance = -1, bestStack, bestVelocity, bestA, bestB;

        for (unsigned int parameterIndex = 0; parameterIndex < totalParameterCount; parameterIndex++) {
            unsigned int offsetParameter = parameterIndex * 3;

            float semblance = semblanceArray[offset + parameterIndex];
            float stack = stackArray[offset + parameterIndex];
            float c = parameterArray[offsetParameter];
            float a = parameterArray[offsetParameter + 1];
            float b = parameterArray[offsetParameter + 2];

            if (semblance > bestSemblance) {
                bestSemblance = semblance;
                bestStack = stack;
                bestVelocity = 2 / sqrt(c);
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
