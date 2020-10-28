#pragma once

#include <cuda.h>

__global__
void computeSemblancesForZeroOffsetCommonReflectionSurface(
    const float *samples,
    const float *midpoint,
    const float *halfoffsetSquared,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    float m0,
    float dtInSeconds,
    int tauIndexDisplacement,
    int windowSize,
    /* Parameter arrays */
    const float *parameterArray,
    const float *nArray,
    unsigned int totalNCount,
    /* Output arrays */
    float *semblanceArray,
    float *stackArray
);
