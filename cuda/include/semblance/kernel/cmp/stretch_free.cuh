#pragma once

#include <cuda.h>

__global__
void computeSemblancesForCommonMidPoint(
    const float *samples,
    const float *halfoffsetSquared,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
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
