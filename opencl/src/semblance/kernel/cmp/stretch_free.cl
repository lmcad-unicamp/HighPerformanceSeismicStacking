#include "common/include/gpu/interface.h"

__kernel
void computeSemblancesForCommonMidPoint(
    __global float *samples,
    __global float *halfoffsetSquared,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    /* Parameter arrays */
    __global float *parameterArray,
    __global float *nArray,
    unsigned int totalNCount,
    /* Output arrays */
    __global float *semblanceArray,
    __global float *stackArray
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

    unsigned int sampleIndex = threadIndex / totalNCount;
    unsigned int nIndex = threadIndex % totalNCount;

    if (sampleIndex < samplesPerTrace) {
        float semblance = 0;
        float stack = 0;

        int n = (int) nArray[nIndex];
        int sampleIndex_n = ((int) sampleIndex) - n;

        if (sampleIndex_n >= 0 && sampleIndex_n < samplesPerTrace) {

            float t0_n = ((float) sampleIndex_n) * dtInSeconds;

            float v_n = parameterArray[sampleIndex_n];
            float c_n = 4.0f / (v_n * v_n);

            float numeratorComponents[MAX_WINDOW_SIZE];
            float denominatorSum = 0;
            float linearSum = 0;

            RESET_SEMBLANCE_NUM_COMP(numeratorComponents, MAX_WINDOW_SIZE);

            unsigned int usedCount = 0;

            for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
                float h_sq = halfoffsetSquared[traceIndex];
                const float *traceSamples = samples + traceIndex * samplesPerTrace;

                float t_n = sqrt(t0_n * t0_n + c_n * h_sq);

                float t = t_n + ((float) n) * dtInSeconds;

                COMPUTE_SEMBLANCE(t, dtInSeconds, samplesPerTrace, tauIndexDisplacement, windowSize, numeratorComponents, linearSum, denominatorSum, usedCount);
            }

            REDUCE_SEMBLANCE_STACK(numeratorComponents, linearSum, denominatorSum, windowSize, usedCount, semblance, stack);
        }

        unsigned int offset = sampleIndex * totalNCount + nIndex;
        semblanceArray[offset] = semblance;
        stackArray[offset] = stack;
    }
}
