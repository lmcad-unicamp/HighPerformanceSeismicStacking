#include "common/include/gpu/interface.h"

__kernel
void buildParameterArrayForZeroOffsetCommonReflectionSurface(
    __global __write_only float* parameterArray,
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
    unsigned int threadIndex = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if (threadIndex < totalParameterCount) {
        unsigned int idxVelocity = (threadIndex / (countA * countB)) % countVelocity;
        unsigned int idxA = (threadIndex / countB) % countA;
        unsigned int idxB = threadIndex % countB;

        float v = minVelocity + ((float) idxVelocity) * incrementVelocity;
        float a = minA + ((float) idxA) * incrementA;
        float b = minB + ((float) idxB) * incrementB;

        unsigned int offset = 3 * threadIndex;
        parameterArray[offset] = 4.0f / (v * v);
        parameterArray[offset + 1] = a;
        parameterArray[offset + 2] = b;
    }
}

__kernel
void computeSemblancesForZeroOffsetCommonReflectionSurface(
    __global __read_only float *samples,
    __global __read_only float *midpoint,
    __global __read_only float *halfoffsetSquared,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float m0,
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
    unsigned int parameterOffset = 3 * parameterIndex;

    if (sampleIndex < samplesPerTrace) {
        float semblance = 0;
        float stack = 0;

        float t0 = ((float) sampleIndex) * dtInSeconds;

        float c = parameterArray[parameterOffset];
        float a = parameterArray[parameterOffset + 1];
        float b = parameterArray[parameterOffset + 2];

        float numeratorComponents[MAX_WINDOW_SIZE];
        float denominatorSum = 0;
        float linearSum = 0;

        for (unsigned int i = 0; i < MAX_WINDOW_SIZE; i++) {
            numeratorComponents[i] = 0;
        }

        unsigned int usedCount = 0;

        for (unsigned int traceIndex = 0; traceIndex < traceCount; traceIndex++) {
            float m = midpoint[traceIndex];
            float h_sq = halfoffsetSquared[traceIndex];
            __global __read_only float *traceSamples = samples + traceIndex * samplesPerTrace;

            float dm = m - m0;
            float tmp = t0 + a * dm;
            tmp = tmp * tmp + b * dm * dm + c * h_sq;

            if (tmp >= 0) {
                float tIndex = sqrt(tmp)/ dtInSeconds;
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

        unsigned int offset = sampleIndex * totalParameterCount + parameterIndex;
        semblanceArray[offset] = semblance;
        stackArray[offset] = stack;
    }
}

__kernel
void selectBestSemblancesForZeroOffsetCommonReflectionSurface(
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
                bestVelocity = 2.0f / sqrt(c);
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
