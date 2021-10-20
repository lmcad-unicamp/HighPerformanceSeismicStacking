#! /bin/bash

# Commit and branch to be used for the experiment
HEAD_COMMIT_ID="9027a04"

# Personal clapp's source directory
CLAPP_SOURCE_PATH="/home/gustavociotto/clap"
CLAPP_CLUSTER="clapp cluster"

# A cluster's name
CLUSTER_NAME="spits-cluster"

# Instance type of the worker nodes
AWS_INSTANCE="g4dn.xlarge"
# Number of worker nodes
TASK_COUNT="32"
# Number of GPUs per node
NW="1"

# Available frameworks: cuda or opencl
FRAMEWORK="cuda"

# Root and NFS directories inside EC2's instance
PROJECT_ROOT="/home/ubuntu/HighPerformanceSeismicStacking"
NFS_MOUNT_POINT="/home/ubuntu/nfs"

# Directory where results are going to be stored
DATA_DIR="${NFS_MOUNT_POINT}/data"

# SPITS's jobs directory
SPITS_JOB_PATH="/home/ubuntu/spits-jobs"

# SPITS compiled binaries' names
SPITS_GREEDY_BIN="spitz_linear_search"
SPITS_DE_BIN="spitz_de"

# Parameters for DE algorithm
GENERATIONS="32"
POPULATION_SIZE="32"

# Thread/block count for worker nodes
THREAD_COUNT="64"

KERNEL_PATH="${PROJECT_ROOT}/opencl/src/semblance/kernel"

if [[ ${AWS_INSTANCE} == *"g4dn"* ]]; then
    COMPUTE_CAPABILITY="75"
elif [[ ${AWS_INSTANCE} == *"p3"* ]]; then
    COMPUTE_CAPABILITY="70"
elif [[ ${AWS_INSTANCE} == *"p2"* ]]; then
    COMPUTE_CAPABILITY="37"
fi

if [ "${FRAMEWORK}" == "opencl" ]; then
    KERNEL_ARGS="--kernel-path ${KERNEL_PATH}"
fi

function build_cluster_common_command {
    JOB_ID=${1}

    echo ${CLAPP_CLUSTER} start ${CLUSTER_NAME} --extra \
            compute_capability=${COMPUTE_CAPABILITY} \
            commit_id=${HEAD_COMMIT_ID} \
            instance_type=${AWS_INSTANCE} \
            task_count=${TASK_COUNT} \
            jobid=${JOBID} \
            nfs_mount_point=${NFS_MOUNT_POINT} \
            project_root=${PROJECT_ROOT} \
            framework=${FRAMEWORK} \
            tmargs="\"--nw=${NW}\""
}

function greedy_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}
    THREAD_COUNT=${3}

    DATE=$(date '+%Y%m%d_%H%M')
    JOBID="${HEAD_COMMIT_ID}-${AWS_INSTANCE}-${NW}-${TASK_COUNT}-${FRAMEWORK}-${DATA_NAME}-cmp-greedy-${DATE}"

    TEST_OUTPUT_DIR="${SPITS_JOB_PATH}/${JOBID}"

    COMMAND=$(build_cluster_common_command ${JOBID})

    SPITS_ARGS="'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 1024 \
        --traveltime cmp \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 \
        ${KERNEL_ARGS}'"

    SPITS_ARGS=$(echo ${SPITS_ARGS} | xargs)

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'"
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'" 2>&1
}

function greedy_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}
    THREAD_COUNT=${3}

    DATE=$(date '+%Y%m%d_%H%M')
    JOBID="${HEAD_COMMIT_ID}-${AWS_INSTANCE}-${NW}-${TASK_COUNT}-${FRAMEWORK}-${DATA_NAME}-zocrs-greedy-${DATE}"

    TEST_OUTPUT_DIR="${SPITS_JOB_PATH}/${JOBID}"

    COMMAND=$(build_cluster_common_command ${JOBID})

    SPITS_ARGS="'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 100 10 10 \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 \
        ${KERNEL_ARGS}'"

    SPITS_ARGS=$(echo ${SPITS_ARGS} | xargs)

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'"
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'" 2>&1
}

function greedy_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}
    THREAD_COUNT=${3}

    DATE=$(date '+%Y%m%d_%H%M')
    JOBID="${HEAD_COMMIT_ID}-${AWS_INSTANCE}-${NW}-${TASK_COUNT}-${FRAMEWORK}-${DATA_NAME}-oct-greedy-${DATE}"

    TEST_OUTPUT_DIR="${SPITS_JOB_PATH}/${JOBID}"

    COMMAND=$(build_cluster_common_command ${JOBID})

    SPITS_ARGS="'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --granularity 100 100 \
        --traveltime oct \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 \
        ${KERNEL_ARGS}'"

    SPITS_ARGS=$(echo ${SPITS_ARGS} | xargs)

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'"
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'" 2>&1
}

function de_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}
    THREAD_COUNT=${3}

    DATE=$(date '+%Y%m%d_%H%M')
    JOBID="${HEAD_COMMIT_ID}-${AWS_INSTANCE}-${NW}-${TASK_COUNT}-${FRAMEWORK}-${DATA_NAME}-cmp-de-${DATE}"

    TEST_OUTPUT_DIR="${SPITS_JOB_PATH}/${JOBID}"

    COMMAND=$(build_cluster_common_command ${JOBID})

    SPITS_ARGS="'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime cmp \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 \
        ${KERNEL_ARGS}'"

    SPITS_ARGS=$(echo ${SPITS_ARGS} | xargs)

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'"
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'" 2>&1
}

function de_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}
    THREAD_COUNT=${3}

    DATE=$(date '+%Y%m%d_%H%M')
    JOBID="${HEAD_COMMIT_ID}-${AWS_INSTANCE}-${NW}-${TASK_COUNT}-${FRAMEWORK}-${DATA_NAME}-zocrs-de-${DATE}"

    TEST_OUTPUT_DIR="${SPITS_JOB_PATH}/${JOBID}"

    COMMAND=$(build_cluster_common_command ${JOBID})

    SPITS_ARGS="--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 \
        ${KERNEL_ARGS}"

    SPITS_ARGS=$(echo ${SPITS_ARGS} | xargs)

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'"
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'" 2>&1
}

function de_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}
    THREAD_COUNT=${3}

    DATE=$(date '+%Y%m%d_%H%M')
    JOBID="${HEAD_COMMIT_ID}-${AWS_INSTANCE}-${NW}-${TASK_COUNT}-${FRAMEWORK}-${DATA_NAME}-oct-de-${DATE}"

    TEST_OUTPUT_DIR="${SPITS_JOB_PATH}/${JOBID}"

    COMMAND=$(build_cluster_common_command ${JOBID})

    SPITS_ARGS="'--aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime oct \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 \
        ${KERNEL_ARGS}'"

    SPITS_ARGS=$(echo ${SPITS_ARGS} | xargs)

    echo ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'"
    ${COMMAND} spits_bin="${SPITS_DE_BIN}" spits_args="'${SPITS_ARGS}'" 2>&1
}

. ${CLAPP_SOURCE_PATH}/clap-env/bin/activate

## Uncomment the following tests to build a cluster and execute it

# de_zero_offset_reflection_surface  "simple-synthetic" "0" ${THREAD_COUNT}
# de_zero_offset_reflection_surface  "fold2000" "90" ${THREAD_COUNT}
# de_offset_continuation_trajectory "simple-synthetic" "0" ${THREAD_COUNT}
# de_offset_continuation_trajectory "fold2000" "90" ${THREAD_COUNT}
