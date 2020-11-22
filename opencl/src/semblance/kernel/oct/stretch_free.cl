#include "opencl/src/semblance/kernel/oct/common.cl"

__kernel
void computeSemblancesForOffsetContinuationTrajectory(
    __global __read_only float *samples,
    __global __read_only float *midpoint,
    __global __read_only float *halfoffset,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float apm,
    float m0,
    float h0,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    /* Parameter arrays */
    __global __read_only float *parameterArray,
    __global __read_only float *nArray,
    unsigned int totalNCount,
    /* Output arrays */
    __global __write_only float* notUsedCountArray,
    __global __write_only float *semblanceArray,
    __global __write_only float *stackArray
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

    unsigned int sampleIndex = threadIndex / totalNCount;
    unsigned int nIndex = threadIndex % totalNCount;

    if (sampleIndex < samplesPerTrace) {
        enum gpu_error_code errorCode;

        float semblance = 0;
        float stack = 0;

        int n = (int) nArray[nIndex];
        int sampleIndex_n = ((int) sampleIndex) - n;

        if (sampleIndex_n >= 0 && sampleIndex_n < samplesPerTrace) {
            float mh, t_n;

            float t0_n = ((float) sampleIndex_n) * dtInSeconds;

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
                    float t = t_n + ((float) n) * dtInSeconds;
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

__kernel
void selectTracesForOffsetContinuationTrajectory(
    __global __read_only float *midpointArray,
    __global __read_only float *halfoffsetArray,
    __global __read_only float *parameterArray,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float dtInSeconds,
    float apm,
    float m0,
    float h0,
    __global __write_only unsigned char* usedTraceMaskArray
) {
    unsigned int traceIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (traceIndex < traceCount) {

        float m, h, mh;
        m = midpointArray[traceIndex];
        h = halfoffsetArray[traceIndex];

        usedTraceMaskArray[traceIndex] = 0;

        for (unsigned int sampleIndex = 0; !usedTraceMaskArray[traceIndex] && sampleIndex < samplesPerTrace; sampleIndex++) {

            float t0 = ((float) sampleIndex) * dtInSeconds;

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
