#pragma once

#include <cuda.h>
#include <curand_kernel.h>

__global__
void setupRandomSeed(
    curandState *state,
    unsigned int seed
);

__global__
void startPopulations(
    float* x,
    const float* min,
    const float* max,
    curandState *st,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters
);

__global__
void mutatePopulations(
    float* v,
    const float* x,
    const float* min,
    const float* max,
    curandState *st,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters
);

__global__
void crossoverPopulations(
    float* u,
    const float* x,
    const float* v,
    curandState *st,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters
);

__global__
void nextGeneration(
    float* x,
    float* fx,
    const float* u,
    const float* fu,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters,
    unsigned int numberOfCommonResults
);
