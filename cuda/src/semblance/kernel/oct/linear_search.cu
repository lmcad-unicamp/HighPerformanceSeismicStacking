#include "cuda/include/semblance/kernel/oct/common.cuh"
#include "cuda/include/semblance/kernel/oct/linear_search.cuh"

__global__
void buildParameterArrayForOffsetContinuationTrajectory(
    float* parameterArray,
    float minVelocity,
    float incrementVelocity,
    unsigned int countVelocity,
    float minSlope,
    float incrementSlope,
    unsigned int countSlope,
    unsigned int totalParameterCount
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIndex < totalParameterCount) {
        unsigned int idxVelocity = (threadIndex / countSlope) % countVelocity;
        unsigned int idxSlope = threadIndex % countSlope;

        float v = minVelocity + static_cast<float>(idxVelocity) * incrementVelocity;
        float slope = minSlope + static_cast<float>(idxSlope) * incrementSlope;

        unsigned int offset = 2 * threadIndex;
        parameterArray[offset] = 4.0f / (v * v);
        parameterArray[offset + 1] = slope;
    }
}

__global__
void computeSemblancesForOffsetContinuationTrajectory(
    const float *samples,
    const float *midpoint,
    const float *halfoffset,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float apm,
    float m0,
    float h0,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    /* Parameter arrays */
    const float *parameterArray,
    unsigned int totalParameterCount,
    /* Output arrays */
    float* notUsedCountArray,
    float *semblanceArray,
    float *stackArray
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int sampleIndex = threadIndex / totalParameterCount;
    unsigned int parameterIndex = threadIndex % totalParameterCount;
    unsigned int parameterOffset = 2 * parameterIndex;

    if (sampleIndex < samplesPerTrace) {
        enum gpu_error_code errorCode;

        float semblance = 0;
        float stack = 0;
        float mh, t;

        float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

        float c = parameterArray[parameterOffset];
        float slope = parameterArray[parameterOffset + 1];

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

        unsigned int offset = sampleIndex * totalParameterCount + parameterIndex;
        semblanceArray[offset] = semblance;
        stackArray[offset] = stack;
    }
}

__global__
void selectBestSemblancesForOffsetContinuationTrajectory(
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

        float bestSemblance = -1, bestStack, bestVelocity, bestSlope;

        for (unsigned int parameterIndex = 0; parameterIndex < totalParameterCount; parameterIndex++) {
            unsigned int offsetParameter = parameterIndex * 2;

            float semblance = semblanceArray[offset + parameterIndex];
            float stack = stackArray[offset + parameterIndex];
            float c = parameterArray[offsetParameter];
            float slope = parameterArray[offsetParameter + 1];

            if (semblance > bestSemblance) {
                bestSemblance = semblance;
                bestStack = stack;
                bestVelocity = 2 / sqrt(c);
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
void selectTracesForOffsetContinuationTrajectory(
    const float *midpointArray,
    const float *halfoffsetArray,
    const float *parameterArray,
    unsigned int parameterCount,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
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

            for (unsigned int parameterIndex = 0; !usedTraceMaskArray[traceIndex] && parameterIndex < parameterCount; parameterIndex++) {
                unsigned int parameterOffset = 2 * parameterIndex;

                float c = parameterArray[parameterOffset];
                float slope = parameterArray[parameterOffset + 1];

                enum gpu_error_code errorCode = computeDisplacedMidpoint(h, h0, t0, m0, c, slope, &mh);

                usedTraceMaskArray[traceIndex] = (errorCode == NO_ERROR && fabs(m - mh) <= apm);
            }
        }
    }
}