#include "common/include/gpu/interface.h"

__kernel
void buildParameterArrayForCommonMidPoint(
    __global float* parameterArray,
    float minVelocity,
    float increment,
    unsigned int totalParameterCount
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (threadIndex < totalParameterCount) {
        float v = minVelocity + ((float) threadIndex) * increment;
        parameterArray[threadIndex] = 4.0f / (v * v);
    }
}

__kernel
void computeSemblancesForCommonMidPoint(
    __global __read_only float *samples,
    __global __read_only float *halfoffsetSquared,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    /* Parameter arrays */
    __global __read_only float *parameterArray,
    unsigned int totalParameterCount,
    /* Output arrays */
    __global __write_only float *semblanceArray,
    __global __write_only float *stackArray
) {
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

    unsigned int sampleIndex = threadIndex / totalParameterCount;
    unsigned int parameterIndex = threadIndex % totalParameterCount;

    if (sampleIndex < samplesPerTrace) {
        float semblance = 0;
        float stack = 0;

        float t0 = ((float) sampleIndex) * dtInSeconds;

        float c = parameterArray[parameterIndex];

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        for (unsigned int i = 0; i < MAX_WINDOW_SIZE; i++) {
            numeratorComponents[i] = 0;
        }

        unsigned int usedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            float h_sq = halfoffsetSquared[traceIndex];
            __global __read_only float *traceSamples = samples + traceIndex * samplesPerTrace;

            float t = sqrt(t0 * t0 + c * h_sq);

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

        if (usedCount > 0) {
            float sumNumerator = 0;
            for (int w = 0; w < windowSize; w++) {
                sumNumerator += numeratorComponents[w] * numeratorComponents[w];
            }

            semblance = sumNumerator / (usedCount * denominatorSum);
            stack = linearSum / (usedCount * windowSize);
        }

        unsigned int offset = sampleIndex * totalParameterCount + parameterIndex;
        semblanceArray[offset] = semblance;
        stackArray[offset] = stack;
    }
}

__kernel
void selectBestSemblancesForCommonMidPoint(
    __global __read_only float *semblanceArray,
    __global __read_only float *stackArray,
    __global __read_only float *parameterArray,
    unsigned int totalParameterCount,
    unsigned int samplesPerTrace,
    __global __write_only float *resultArray
) {
    unsigned int sampleIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

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
                bestVelocity = 2.0f / sqrt(c);
            }
        }

        resultArray[sampleIndex] = bestSemblance;
        resultArray[samplesPerTrace + sampleIndex] = bestStack;
        resultArray[2 * samplesPerTrace + sampleIndex] = bestVelocity;
    }
}