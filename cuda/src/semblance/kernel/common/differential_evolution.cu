#include "common/include/gpu/interface.h"
#include "cuda/include/semblance/kernel/common/differential_evolution.cuh"

__global__
void setupRandomSeed(
    curandState *state,
    unsigned int seed,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace
) {
    unsigned int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;

    if (sampleIndex < samplesPerTrace) {
        curand_init(seed, threadIndex, 0, &state[threadIndex]);
    }
}

__global__
void startPopulations(
    float* x,
    const float* min,
    const float* max,
    curandState *st,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters
) {
    unsigned int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;
    unsigned int seedIndex = threadIndex;

    if (sampleIndex < samplesPerTrace) {
        unsigned int offset = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfParameters;
        for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
            float ratio = curand_uniform(&st[seedIndex]);
            x[offset + parameterIndex] = min[parameterIndex] + ratio * (max[parameterIndex] - min[parameterIndex]);
        }
    }
}

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
) {
    unsigned int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;
    unsigned int seedIndex = threadIndex;

    if (sampleIndex < samplesPerTrace) {
        unsigned int popIndex = sampleIndex * individualsPerPopulation * numberOfParameters;
        unsigned int offset = popIndex + individualIndex * numberOfParameters;

        float p1, p2, p3;
        unsigned int r1, r2, r3;
        for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
            p1 = curand_uniform(&st[seedIndex]);
            p2 = curand_uniform(&st[seedIndex]);
            p3 = curand_uniform(&st[seedIndex]);

            r1 = popIndex + 
                static_cast<unsigned int>(p1 * static_cast<float>(individualsPerPopulation - 1)) * numberOfParameters + 
                parameterIndex;

            r2 = popIndex + 
                static_cast<unsigned int>(p2 * static_cast<float>(individualsPerPopulation - 1)) * numberOfParameters + 
                parameterIndex;

            r3 = popIndex + 
                static_cast<unsigned int>(p3 * static_cast<float>(individualsPerPopulation - 1)) * numberOfParameters + 
                parameterIndex;

            float newIndividual = x[r1] + F_FAC * (x[r2] - x[r3]);
            newIndividual = fminf(newIndividual, max[parameterIndex]);
            newIndividual = fmaxf(newIndividual, min[parameterIndex]);

            v[offset + parameterIndex] = newIndividual;
        }
    }
}

__global__
void crossoverPopulations(
    float* u,
    const float* x,
    const float* v,
    curandState *st,
    unsigned int individualsPerPopulation,
    unsigned int samplesPerTrace,
    unsigned int numberOfParameters
) {
    unsigned int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;
    unsigned int seedIndex = threadIndex;

    if (sampleIndex < samplesPerTrace) {
        unsigned int l = static_cast<unsigned int>(curand_uniform(&st[seedIndex]) * static_cast<float>(numberOfParameters - 1));
        unsigned int offset = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfParameters;

        for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
            float r = curand_uniform(&st[seedIndex]);
            unsigned int featureIndex = offset + parameterIndex;

            if (r > CR && l != parameterIndex) {
                u[featureIndex] = x[featureIndex];
            }
            else {
                u[featureIndex] = v[featureIndex];
            }
        }
    }
}

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
) {
    unsigned int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int sampleIndex = threadIndex / individualsPerPopulation;
    unsigned int individualIndex = threadIndex % individualsPerPopulation;

    if (sampleIndex < samplesPerTrace) {
        unsigned int popIndex = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfParameters;
        unsigned int fitnessIndex = (sampleIndex * individualsPerPopulation + individualIndex) * numberOfCommonResults;

        if (fu[fitnessIndex] > fx[fitnessIndex]) {
            for (unsigned int parameterIndex = 0; parameterIndex < numberOfParameters; parameterIndex++) {
                unsigned int featureIndex = popIndex + parameterIndex;
                x[featureIndex] = u[featureIndex];
            }

            for (unsigned int resultIndex = 0; resultIndex < numberOfCommonResults; resultIndex++) {
                unsigned int perfIndex = fitnessIndex + resultIndex;
                fx[perfIndex] = fu[perfIndex];
            }
        }
    }
}
