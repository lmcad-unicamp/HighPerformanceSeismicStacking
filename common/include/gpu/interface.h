#ifndef GPU_INTEFACE_H
#define GPU_INTEFACE_H

#include "common/include/capability.h"

#define MAX_PARAMETER_COUNT 3
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WINDOW_SIZE 64

enum gpu_error_code {
    NO_ERROR,
    DIVISION_BY_ZERO,
    NEGATIVE_SQUARED_ROOT,
    INVALID_RANGE,
    INVALID_MODEL
};

typedef struct {
    unsigned int samplesPerTrace;
    int tauIndexDisplacement, windowSize;
    float apm, tau, dtInSeconds;
} gpu_gather_data_t;

typedef struct {
    float m0, t0, h0;
} gpu_reference_point_t;

typedef struct {
    float numeratorComponents[MAX_WINDOW_SIZE];
    float denominatorSum, linearSum;
} gpu_semblance_compute_data_t;

typedef struct {
    float coherence, stacking;
    unsigned long notUsedTrace;
} gpu_semblance_result_t;

typedef struct {
    enum traveltime_t traveltime;
    unsigned int numberOfParameters, numberOfCommonResults;
} gpu_traveltime_data_t;

typedef struct {
    float semblanceParameters[MAX_PARAMETER_COUNT];
    float mh;
    unsigned int numberOfParameters;
} gpu_traveltime_parameter_t;

#endif