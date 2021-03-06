#! /bin/bash

TEST_ID="0447d1c"

CUDA_ROOT=$(dirname ${PWD})
PROJECT_ROOT=$(dirname ${CUDA_ROOT})

FRAMEWORK="cuda"

DATA_DIR="${PROJECT_ROOT}/data"
BIN_DIR="${CUDA_ROOT}/bin"
TEST_DIR="${PROJECT_ROOT}/out/${TEST_ID}"

function greedy_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_INDEX=${3}
    THREAD_COUNT=${4}

    TEST_OUTPUT_DIR="${TEST_DIR}/cmp/greedy/${FRAMEWORK}/${DATA_NAME}_${THREAD_COUNT}_${TEST_INDEX}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 1024 \
        --traveltime cmp \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}

function greedy_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_INDEX=${3}
    THREAD_COUNT=${4}

    TEST_OUTPUT_DIR="${TEST_DIR}/zocrs/greedy/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 100 10 10 \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}

function greedy_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_INDEX=${3}
    THREAD_COUNT=${4}

    TEST_OUTPUT_DIR="${TEST_DIR}/oct/greedy/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    mkdir -p ${TEST_OUTPUT_DIR}

    (set -x; \
    ${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --granularity 100 100 \
        --traveltime oct \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log)
}


THREAD_COUNTS=(32 64 128 256 512 1024)

for THREAD_COUNT in ${THREAD_COUNTS[@]}; do
    #### Common Mid Point
    for i in `seq 1 3`;
    do
        echo "Executing ${i}th iteration - CMP"
        greedy_common_mid_point "fold2000" "90" ${i} ${THREAD_COUNT}
        greedy_common_mid_point "simple-synthetic" "0" ${i} ${THREAD_COUNT}
    done
done

#### Zero Offset Common Reflection Point

# for i in `seq 1 10`;
# do
#     echo "Executing ${i}th iteration - ZOCRS"
#     greedy_zero_offset_reflection_surface "fold2000" "90" ${i}
#     greedy_zero_offset_reflection_surface "simple-synthetic" "0" ${i}
# done

#### Offset Continuation Trajectory

# for i in `seq 1 10`;
# do
#     echo "Executing ${i}th iteration - OCT"
#     greedy_offset_continuation_trajectory "fold2000" "90" ${i}
#     greedy_offset_continuation_trajectory "simple-synthetic" "0" ${i}
# done
