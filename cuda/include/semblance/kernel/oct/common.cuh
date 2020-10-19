#pragma once

#include "common/include/gpu/interface.h"

#include <cuda.h>

__forceinline__
__device__
enum gpu_error_code computeDisplacedMidpoint(float h, float h0, float t0, float m0, float c, float slope, float* mh) {
    float theta, theta_sq, gamma, gamma_sq, tn0_sq, tn0_quad, sqrt_arg;

    theta = t0 * slope;
    gamma = 2 * sqrt(h * h + h0 * h0);
    tn0_sq = t0 * t0 - c * h0 * h0;

    theta_sq = theta * theta;
    gamma_sq = gamma * gamma;
    tn0_quad = tn0_sq * tn0_sq;

    sqrt_arg = tn0_quad * tn0_quad + tn0_quad * theta_sq * gamma_sq + 16 * h * h * h0 * h0 * theta_sq * theta_sq;

    if (sqrt_arg < 0) {
        return NEGATIVE_SQUARED_ROOT;
    }

    sqrt_arg = theta_sq * gamma_sq + 2 * tn0_quad + 2 * sqrt(sqrt_arg);

    if (sqrt_arg == 0) {
        return DIVISION_BY_ZERO;
    }

    *mh = m0 + 2 * theta * (h * h - h0 * h0) / sqrt(sqrt_arg);

    return NO_ERROR;
}

__forceinline__
__device__
enum gpu_error_code computeTime(float h, float h0, float t0, float m0, float m, float mh, float c, float slope, float* out) {
    float tn0_sq, tn_sq, w_sqrt_1, w_sqrt_2, u, sqrt_arg, ah, th;

    tn0_sq = t0 * t0 - c * h0 * h0;

    w_sqrt_1 = (h + h0) * (h + h0) - (mh - m0) * (mh - m0);
    w_sqrt_2 = (h - h0) * (h - h0) - (mh - m0) * (mh - m0);

    if (w_sqrt_1 < 0 || w_sqrt_2 < 0) {
        return NEGATIVE_SQUARED_ROOT;
    }

    u = sqrt(w_sqrt_1) + sqrt(w_sqrt_2);

    th = t0;
    if (fabs(h) > fabs(h0)) {

        if (!u) {
            return DIVISION_BY_ZERO;
        }

        sqrt_arg = c * h * h + 4 * h * h / (u * u) * tn0_sq;

        if (sqrt_arg < 0) {
            return NEGATIVE_SQUARED_ROOT;
        }

        th = sqrt(sqrt_arg);
    }
    else if (fabs(h) < fabs(h0)) {

        if (!h0) {
            return DIVISION_BY_ZERO;
        }

        sqrt_arg = c * h * h + u * u / (4 * h0 * h0) * tn0_sq;

        if (sqrt_arg < 0) {
            return NEGATIVE_SQUARED_ROOT;
        }

        th = sqrt(sqrt_arg);
    }

    tn_sq = th * th - c * h * h;

    if (!th || !tn0_sq) {
        return DIVISION_BY_ZERO;
    }

    ah = (t0 * tn_sq) / (th * tn0_sq) * slope;

    *out = th + ah * (m - mh);

    return NO_ERROR;
}
