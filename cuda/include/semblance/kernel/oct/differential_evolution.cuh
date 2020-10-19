#pragma once

#include <cuda.h>

__global__
void computeSemblancesForOffsetContinuationTrajectory(
    const float *samples,
    const float *midpoint,
    const float *halfoffset,
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
    float* notUsedCountArray,
    const float *x,
    float *fx
);

__global__
void selectBestIndividualsForOffsetContinuationTrajectory(
    const float* x,
    const float* fx,
    float* resultArray,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfCommonResults
);

__global__
void selectTracesForOffsetContinuationTrajectoryAndDifferentialEvolution(
    const float *midpointArray,
    const float *halfoffsetArray,
    const float *x,
    unsigned int traceCount,
    unsigned int samplesPerTrace,
    unsigned int individualsPerPopulation,
    float dtInSeconds,
    float apm,
    float m0,
    float h0,
    unsigned char* usedTraceMaskArray
);
