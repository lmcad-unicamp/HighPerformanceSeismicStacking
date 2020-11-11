#include "opencl/src/semblance/kernel/oct/common.cl"

__kernel
void computeSemblancesForOffsetContinuationTrajectory(
    __global __read_only float *samples,
    __global __read_only float *midpoint,
    __global __read_only float *halfoffset,
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
    __global __write_only float* notUsedCountArray,
    __global __read_only float *x,
    __global __write_only float *fx
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;

    if (sampleIndex < samplesPerTrace) {
        enum gpu_error_code errorCode;

        float semblance = 0;
        float stack = 0;
        float mh, t;

        unsigned int parameterOffset = (sampleIndex * individualsPerPopulation + individualIndex) * 2;

        float t0 = ((float) sampleIndex) * dtInSeconds;

        float v = x[parameterOffset];
        float c = 4.0f / (v * v);
        float slope = x[parameterOffset + 1];

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        for (unsigned int i = 0; i < MAX_WINDOW_SIZE; i++) {
            numeratorComponents[i] = 0;
        }

        unsigned int usedCount = 0, notUsedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            float m = midpoint[traceIndex];
            float h = halfoffset[traceIndex];
            __global __read_only float *traceSamples = samples + traceIndex * samplesPerTrace;

            errorCode = computeDisplacedMidpoint(h, h0, t0, m0, c, slope, &mh);

            if (errorCode != NO_ERROR || fabs(m - mh) > apm) {
                notUsedCount++;
                continue;
            }

            errorCode = computeTime(h, h0, t0, m0, m, mh, c, slope, &t);

            if (errorCode == NO_ERROR) {
                float tIndex = t / dtInSeconds;
                int kIndex = (int) tIndex;
                float dt = tIndex - (float) kIndex;
            
                if ((kIndex - tauIndexDisplacement >= 0) &&
                    (kIndex + tauIndexDisplacement + 1 < (int) samplesPerTrace)) {
                    
                    int k = kIndex - tauIndexDisplacement;
                    float u, y0, y1;
                    
                    y1 = traceSamples[k];
                    
                    for (int j = 0; j < windowSize; j++, k++) {
                        y0 = y1;
                        y1 = traceSamples[k + 1];
                        u = (y1 - y0) * dt + y0;
            
                        numeratorComponents[j] += u;
                        linearSum += u;
                        denominatorSum += u * u;
                    }
            
                    usedCount++;
                }
            }
        }

        if (usedCount > 0) {
            float sumNumerator = 0;
            for (int w = 0; w < windowSize; w++) {
                sumNumerator += numeratorComponents[w] * numeratorComponents[w];
            }

            semblance = sumNumerator / (usedCount * denominatorSum);
            stack = linearSum / (usedCount * windowSize);
        }

        notUsedCountArray[threadIndex] += notUsedCount;

        unsigned int offset = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfCommonResults;
        fx[offset] = semblance;
        fx[offset + 1] = stack;
    }
}

__kernel
void selectBestIndividualsForOffsetContinuationTrajectory(
    __global __read_only float* x,
    __global __read_only float* fx,
    __global __write_only float* resultArray,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfCommonResults
) {
    unsigned int sampleIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

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

__kernel
void selectTracesForOffsetContinuationTrajectory(
    __global __read_only float *midpointArray,
    __global __read_only float *halfoffsetArray,
    __global __read_only float *x,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    unsigned int individualsPerPopulation,
    float dtInSeconds,
    float apm,
    float m0,
    float h0,
    __global unsigned char* usedTraceMaskArray
) {
    unsigned int traceIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (traceIndex < traceCount) {

        float m, h, mh;
        m = midpointArray[traceIndex];
        h = halfoffsetArray[traceIndex];

        for (unsigned int sampleIndex = 0; !usedTraceMaskArray[traceIndex] && sampleIndex < samplesPerTrace; sampleIndex++) {
            float t0 = ((float) sampleIndex) * dtInSeconds;

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
