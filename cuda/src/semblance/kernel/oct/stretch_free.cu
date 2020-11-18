#include "common/include/gpu/interface.h"
#include "cuda/include/semblance/kernel/oct/common.cuh"
#include "cuda/include/semblance/kernel/oct/stretch_free.cuh"

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
    const float *nArray,
    unsigned int totalNCount,
    /* Output arrays */
    float* notUsedCountArray,
    float *semblanceArray,
    float *stackArray
) {
    unsigned int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int sampleIndex = threadIndex / totalNCount;
    unsigned int nIndex = threadIndex % totalNCount;

    if (sampleIndex < samplesPerTrace) {
        enum gpu_error_code errorCode;

        float semblance = 0;
        float stack = 0;

        int n = static_cast<int>(nArray[nIndex]);
        int sampleIndex_n = static_cast<int>(sampleIndex) - n;

        if (sampleIndex_n >= 0 && sampleIndex_n < samplesPerTrace) {
            float mh, t_n;

            float t0_n = static_cast<float>(sampleIndex_n) * dtInSeconds;

            float v_n = parameterArray[sampleIndex_n];
            float c_n = 4.0f / (v_n * v_n);
            float slope_n = parameterArray[samplesPerTrace + sampleIndex_n];

            float numeratorComponents[MAX_WINDOW_SIZE];
            float denominatorSum = 0;
            float linearSum = 0;

            RESET_SEMBLANCE_NUM_COMP(numeratorComponents, MAX_WINDOW_SIZE);

            unsigned int usedCount = 0, notUsedCount = 0;

            for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
                float m = midpoint[traceIndex];
                float h = halfoffset[traceIndex];
                const float *traceSamples = samples + traceIndex * samplesPerTrace;

                errorCode = computeDisplacedMidpoint(h, h0, t0_n, m0, c_n, slope_n, &mh);

                if (errorCode != NO_ERROR || fabs(m - mh) > apm) {
                    notUsedCount++;
                    continue;
                }

                errorCode = computeTime(h, h0, t0_n, m0, m, mh, c_n, slope_n, &t_n);

                if (errorCode == NO_ERROR) {
                    float t = t_n + static_cast<float>(n) * dtInSeconds;
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
        }

        unsigned int offset = sampleIndex * totalNCount + nIndex;
        semblanceArray[offset] = semblance;
        stackArray[offset] = stack;
    }
}

__global__
void selectTracesForOffsetContinuationTrajectory(
    const float *midpointArray,
    const float *halfoffsetArray,
    const float *parameterArray,
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

        usedTraceMaskArray[traceIndex] = 0;

        for (unsigned int sampleIndex = 0; !usedTraceMaskArray[traceIndex] && sampleIndex < samplesPerTrace; sampleIndex++) {

            float t0 = static_cast<float>(sampleIndex) * dtInSeconds;

            unsigned int velocityIndex = sampleIndex;
            unsigned int slopeIndex = samplesPerTrace + sampleIndex;

            float v = parameterArray[velocityIndex];
            float c = 4.0f / (v * v);
            float slope = parameterArray[slopeIndex];

            enum gpu_error_code errorCode = computeDisplacedMidpoint(h, h0, t0, m0, c, slope, &mh);

            usedTraceMaskArray[traceIndex] = (errorCode == NO_ERROR && fabs(m - mh) <= apm);
        }
    }
}
