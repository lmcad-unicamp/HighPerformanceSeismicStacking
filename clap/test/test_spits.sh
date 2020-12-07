#! /bin/bash

TEST_ID="6fe1798"

CLAPP_SOURCE_PATH="/Users/ciotto/Projects/Master/clapp"
CLAPP_CLUSTER="clapp cluster"

CLUSTER_NAME="spits-cluster"

INSTANCE_TYPE="p2.xlarge"
INSTANCE_COMPUTE_CAPABILITY="sm_37"
TASK_COUNT="1"
NW="1"

FRAMEWORK="opencl"

PROJECT_ROOT="/home/ubuntu/HighPerformanceSeismicStacking"
NFS_MOUNT_POINT="/home/ubuntu/nfs"
DATA_DIR="${NFS_MOUNT_POINT}/data"
TEST_DIR="${NFS_MOUNT_POINT}/out/${INSTANCE_TYPE}/${TEST_ID}/${FRAMEWORK}"

SPITS_GREEDY_BIN="spitz_linear_search"
SPITS_DE_BIN="spitz_de"

GENERATIONS="32"
POPULATION_SIZE="32"

KERNEL_PATH="${PROJECT_ROOT}/opencl/src/semblance/kernel"

if [ "${FRAMEWORK}" == "opencl" ]; then
    KERNEL_ARGS="--kernel-path ${KERNEL_PATH}"
fi

function build_cluster_common_command {
    DATA_NAME=${1}
    TEST_OUTPUT_DIR=${2}

    DATE=$(date '+%Y%m%d_%H%M')

    JOBID="${TEST_ID}-${INSTANCE_TYPE}-${FRAMEWORK}-${DATA_NAME}-${DATE}"

    echo ${CLAPP_CLUSTER} start ${CLUSTER_NAME} --extra \
            compute_capability=\"${INSTANCE_COMPUTE_CAPABILITY}\" \
            instance_type="${INSTANCE_TYPE}" \
            task_count=\"${TASK_COUNT}\" \
            jobid=\"${JOBID}\" \
            nfs_mount_point=\"${NFS_MOUNT_POINT}\" \
            test_dir=\"${TEST_OUTPUT_DIR}\" \
            project_root=\"${PROJECT_ROOT}\" \
            framework=\"${FRAMEWORK}\" \
            tmargs="\"--nw=${NW}\""
}

function greedy_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/cmp/greedy/${FRAMEWORK}/${DATA_NAME}"

    COMMAND=$(build_cluster_common_command ${DATA_NAME} ${TEST_OUTPUT_DIR})

    SPITS_ARGS="\"'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 1024 \
        --traveltime cmp \
        --verbose 1 \
        ${KERNEL_ARGS}'\""

    echo ${COMMAND} spits_bin=${SPITS_GREEDY_BIN} spits_args=${SPITS_ARGS}
    ${COMMAND} spits_bin=${SPITS_GREEDY_BIN} spits_args=${SPITS_ARGS} 2>&1
}

function greedy_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/zocrs/greedy/${FRAMEWORK}/${DATA_NAME}"

    COMMAND=$(build_cluster_common_command ${DATA_NAME} ${TEST_OUTPUT_DIR})

    SPITS_ARGS="\"'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 100 10 10 \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --verbose 1 \
        ${KERNEL_ARGS}'\""

    echo ${COMMAND} spits_bin=${SPITS_GREEDY_BIN} spits_args=${SPITS_ARGS}
    ${COMMAND} spits_bin=${SPITS_GREEDY_BIN} spits_args=${SPITS_ARGS} 2>&1
}

function greedy_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/oct/greedy/${FRAMEWORK}/${DATA_NAME}"

    COMMAND=$(build_cluster_common_command ${DATA_NAME} ${TEST_OUTPUT_DIR})

    SPITS_ARGS="\"'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --granularity 100 100 \
        --traveltime oct \
        --verbose 1 \
        ${KERNEL_ARGS}'\""

    echo ${COMMAND} spits_bin=${SPITS_GREEDY_BIN} spits_args=${SPITS_ARGS}
    ${COMMAND} spits_bin=${SPITS_GREEDY_BIN} spits_args=${SPITS_ARGS} 2>&1
}

function de_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/cmp/de/${FRAMEWORK}/${DATA_NAME}"

    COMMAND=$(build_cluster_common_command ${DATA_NAME} ${TEST_OUTPUT_DIR})

    SPITS_ARGS="\"'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime cmp \
        --verbose 1 \
        ${KERNEL_ARGS}'\""

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args=${SPITS_ARGS}
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args=${SPITS_ARGS} 2>&1
}

function de_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/zocrs/de/${FRAMEWORK}/${DATA_NAME}"

    COMMAND=$(build_cluster_common_command ${DATA_NAME} ${TEST_OUTPUT_DIR})

    SPITS_ARGS="\"'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --verbose 1 \
        ${KERNEL_ARGS}'\""

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args=${SPITS_ARGS}
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args=${SPITS_ARGS} 2>&1
}

function de_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}

    TEST_OUTPUT_DIR="${TEST_DIR}/oct/de/${FRAMEWORK}/${DATA_NAME}"

    COMMAND=$(build_cluster_common_command ${DATA_NAME} ${TEST_OUTPUT_DIR})

    SPITS_ARGS="\"'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime oct \
        --verbose 1 \
        ${KERNEL_ARGS}'\""

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args=${SPITS_ARGS}
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args=${SPITS_ARGS} 2>&1
}

. ${CLAPP_SOURCE_PATH}/clap-env/bin/activate

greedy_zero_offset_reflection_surface  "simple-synthetic" "0"
