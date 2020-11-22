#! /bin/bash

TEST_ID="e948eb2b"

OPENCL_ROOT=$(dirname ${PWD})
PROJECT_ROOT=$(dirname ${OPENCL_ROOT})

FRAMEWORK="opencl"

DATA_DIR="${PROJECT_ROOT}/data"
BIN_DIR="${OPENCL_ROOT}/bin"
TEST_DIR="${PROJECT_ROOT}/out/${TEST_ID}"

KERNEL_PATH="${OPENCL_ROOT}/src/semblance/kernel"

function greedy_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/cmp/greedy/${FRAMEWORK}/${DATA_NAME}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 1024 \
        --traveltime cmp \
        --kernel-path ${KERNEL_PATH} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}

function greedy_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/zocrs/greedy/${FRAMEWORK}/${DATA_NAME}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 100 10 10 \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --kernel-path ${KERNEL_PATH} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}

function greedy_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/oct/greedy/${FRAMEWORK}/${DATA_NAME}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --granularity 100 100 \
        --traveltime oct \
        --kernel-path ${KERNEL_PATH} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}

#### Common Mid Point

greedy_common_mid_point "fold2000" "90"
greedy_common_mid_point "simple-synthetic" "0"

#### Zero Offset Common Reflection Point

greedy_zero_offset_reflection_surface "fold2000" "90"
greedy_zero_offset_reflection_surface "simple-synthetic" "0"

#### Offset Continuation Trajectory

greedy_offset_continuation_trajectory "fold2000" "90"
greedy_offset_continuation_trajectory "simple-synthetic" "0"
