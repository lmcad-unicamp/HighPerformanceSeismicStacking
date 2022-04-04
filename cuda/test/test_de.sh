#! /bin/bash

TEST_ID="<add a test id>"
GPU="gtxtitan"

CUDA_ROOT=$(dirname ${PWD})
PROJECT_ROOT=$(dirname ${CUDA_ROOT})

FRAMEWORK="cuda"

DATA_DIR="${PROJECT_ROOT}/data"
BIN_DIR="${CUDA_ROOT}/bin"
TEST_DIR="${PROJECT_ROOT}/out/${TEST_ID}_${GPU}"

GENERATIONS="32"
POPULATION_SIZE="32"

function de_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_INDEX=${3}
    THREAD_COUNT=${4}

    TEST_OUTPUT_DIR="${TEST_DIR}/cmp/de/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_de \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime cmp \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}

function de_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_INDEX=${3}
    THREAD_COUNT=${4}

    TEST_OUTPUT_DIR="${TEST_DIR}/zocrs/de/${FRAMEWORK}/${DATA_NAME}_${THREAD_COUNT}_${TEST_INDEX}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_de \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}

function de_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_INDEX=${3}
    THREAD_COUNT=${4}

    TEST_OUTPUT_DIR="${TEST_DIR}/oct/de/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_de \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime oct \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}

## Example 1: Run Common Mid Point 10 times
# THREAD_COUNT=64
# for i in `seq 1 10`;
# do
#    echo "Executing ${i}th iteration - CMP"
#    de_common_mid_point "fold2000" "90" ${i} ${THREAD_COUNT}
#    de_common_mid_point "simple-synthetic" "0" ${i} ${THREAD_COUNT}
# done

## Example 2: Run Zero Offset Common Reflection Point 10 times
# THREAD_COUNT=64
# for i in `seq 1 10`;
# do
#     echo "Executing ${i}th iteration - ZOCRS"
#     de_zero_offset_reflection_surface "fold2000" "90" ${i} ${THREAD_COUNT}
#     de_zero_offset_reflection_surface "simple-synthetic" "0" ${i} ${THREAD_COUNT}
# done

## Example 3: Run Offset Continuation Trajectory 10 times
# THREAD_COUNT=64
# for i in `seq 1 10`;
# do
#     echo "Executing ${i}th iteration - OCT"
#     de_offset_continuation_trajectory "fold2000" "90" ${i} ${THREAD_COUNT}
#     de_offset_continuation_trajectory "simple-synthetic" "0" ${i} ${THREAD_COUNT}
# done

## Example 4: Assess different threads per block
# THREAD_COUNTS=(32 64 128 256 512 1024)
# for THREAD_COUNT in ${THREAD_COUNTS[@]}; do
#     for i in `seq 1 3`;
#     do
#         echo "Executing ${i}th iteration - ZOCRS"
#         de_zero_offset_reflection_surface "fold2000" "90" ${i} ${THREAD_COUNT}
#         de_zero_offset_reflection_surface "simple-synthetic" "0" ${i} ${THREAD_COUNT}
#     done
# done
