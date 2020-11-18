#pragma once

#include "common/include/capability.h"

#define F_FAC 0.85f
#define CR 0.5f

#define MAX_PARAMETER_COUNT 3
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WINDOW_SIZE 32

enum gpu_error_code {
    NO_ERROR,
    DIVISION_BY_ZERO,
    NEGATIVE_SQUARED_ROOT,
    INVALID_RANGE,
    INVALID_MODEL
};

#define RESET_SEMBLANCE_NUM_COMP(numeratorComponents, componentCount) do { \
    for (unsigned int _i = 0; _i < componentCount; _i++) { \
        numeratorComponents[_i] = 0; \
    } \
} while (0)

#define COMPUTE_SEMBLANCE(t, dtInSeconds, samplesPerTrace, tauIndexDisplacement, windowSize, numeratorComponents, linearSum, denominatorSum, usedCount) do { \
    float tIndex = t / dtInSeconds; \
    int kIndex = (int) tIndex; \
    float dt = tIndex - (float) kIndex; \
    if ((kIndex - tauIndexDisplacement >= 0) && (kIndex + tauIndexDisplacement + 1 < (int) samplesPerTrace)) { \
        int k = kIndex - tauIndexDisplacement; \
        float u, y0, y1; \
        y1 = traceSamples[k]; \
        for (int j = 0; j < windowSize; j++, k++) { \
            y0 = y1; \
            y1 = traceSamples[k + 1]; \
            u = (y1 - y0) * dt + y0; \
            numeratorComponents[j] += u; \
            linearSum += u; \
            denominatorSum += u * u; \
        } \
        usedCount++; \
    } \
} while (0)

#define REDUCE_SEMBLANCE_STACK(numeratorComponents, linearSum, denominatorSum, windowSize, usedCount, semblance, stack) do { \
    if (usedCount > 0) { \
        float _sumNumerator = 0; \
        for (int _w = 0; _w < windowSize; _w++) { \
            _sumNumerator += numeratorComponents[_w] * numeratorComponents[_w]; \
        } \
        semblance = _sumNumerator / (usedCount * denominatorSum); \
        stack = linearSum / (usedCount * windowSize); \
    } \
} while (0)
