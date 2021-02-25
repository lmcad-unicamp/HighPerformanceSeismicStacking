#pragma once

#include <cuda.h>

__global__
void buildParameterArrayForZeroOffsetCommonReflectionSurface(
    float* parameterArray,
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
);

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
    unsigned int totalParameterCount,
    /* Output arrays */
    float *semblanceArray,
    float *stackArray
);

__global__
void selectBestSemblancesForZeroOffsetCommonReflectionSurface(
    const float *semblanceArray,
    const float *stackArray,
    const float *parameterArray,
    unsigned int totalParameterCount,
    unsigned int samplesPerTrace,
    float *resultArray
);
