#! /bin/bash

# Commit for experiment identification only
HEAD_COMMIT_ID="caa5ee9"

# Personal clapp's source directory
CLAPP_SOURCE_PATH="/home/gustavociotto/clap"
CLAPP_CLUSTER="clapp cluster"

# EFS mount point
NFS_MOUNT_POINT="/home/ubuntu/nfs"

# The SPITS' log directory
SPITS_JOB_PATH="/home/ubuntu/spits-jobs"

set -e

function mkdir_node {
    NODE_ID=${1}
    DIR_TO_BE_CREATED=${2}

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="mkdir -p ${DIR_TO_BE_CREATED}" 2>&1)
}

function copy_node {
    NODE_ID=${1}
    FROM_DIR_PATH=${2}
    TO_DIR_PATH=${3}

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="cp -r ${FROM_DIR_PATH} ${TO_DIR_PATH}" 2>&1)
}

function resume_cluster {
    CLUSTER_ID=${1}

    (set -x; \
    ${CLAPP_CLUSTER} resume ${CLUSTER_ID} 2>&1)
}

function stop_cluster {
    CLUSTER_ID=${1}

    (set -x; \
    ${CLAPP_CLUSTER} stop ${CLUSTER_ID} 2>&1)
}

function copy_from_taskmanager {
    STARTING_NODE=${1}
    TASK_MANAGER_NODE_IDX=${2}
    TASK_MANAGER_OUTPUT_DIR=${3}

    TASK_MANAGER_NODE_ID="node-$(expr $STARTING_NODE + $TASK_MANAGER_NODE_IDX)"

    TASK_MANAGER_NODE_OUT_DIR="${TASK_MANAGER_OUTPUT_DIR}/${TASK_MANAGER_NODE_IDX}"

    mkdir_node ${TASK_MANAGER_NODE_ID} ${TASK_MANAGER_NODE_OUT_DIR}
    copy_node ${TASK_MANAGER_NODE_ID} ${SPITS_JOB_PATH} ${TASK_MANAGER_NODE_OUT_DIR}
}

function fetch_and_stop {
    AWS_INSTANCE=${1}
    DATA_NAME=${2}
    TRAVELTIME=${3}
    COMPUTE_ALGORITHM=${4}
    FRAMEWORK=${5}
    CLUSTER_ID=${6}
    NODE_COUNT=${7}
    STARTING_NODE=${8}
    TEST_INDEX=${9}

    TEST_ID="${HEAD_COMMIT_ID}_spits_${AWS_INSTANCE}"
    TEST_DIR="${NFS_MOUNT_POINT}/out/${TEST_ID}/${TRAVELTIME}/${COMPUTE_ALGORITHM}/${FRAMEWORK}/${DATA_NAME}/${NODE_COUNT}-node/${TEST_INDEX}/"

    JOB_MANAGER_NODE_ID="node-${STARTING_NODE}"
    JOB_MANAGER_OUTPUT_DIR="${TEST_DIR}/jobmanager"

    TASK_MANAGER_OUTPUT_DIR="${TEST_DIR}/taskmanager"

    resume_cluster ${CLUSTER_ID}

    mkdir_node ${JOB_MANAGER_NODE_ID} ${JOB_MANAGER_OUTPUT_DIR}
    copy_node ${JOB_MANAGER_NODE_ID} ${SPITS_JOB_PATH} ${JOB_MANAGER_OUTPUT_DIR} &

    PIDS="$!"
    for TASK_MANAGER_NODE_IDX in `seq 1 ${NODE_COUNT}`; do
        copy_from_taskmanager $STARTING_NODE $TASK_MANAGER_NODE_IDX $TASK_MANAGER_OUTPUT_DIR &
        PIDS="${PIDS} $!"
    done

    wait ${PIDS}
    stop_cluster ${CLUSTER_ID}
}

. ${CLAPP_SOURCE_PATH}/clap-env/bin/activate

# Use the following example to fetch logs and data from all nodes.
# fetch_and_stop AWS_INSTANCE DATA_NAME TRAVELTIME COMPUTE_ALGORITHM FRAMEWORK CLUSTER_ID NODE_COUNT STARTING_NODE TEST_INDEX
# AWS_INSTANCE, DATA_NAME, TRAVELTIME, COMPUTE_ALGORITHM, FRAMEWORK, and TEST_INDEX are meant to identify the experiment
# CLUSTER_ID, NODE_COUNT and STARTING_NODE are meant to identify cluster and nodes. All nodes from STARTING_NODE to
#   STARTING_NODE + NODE_COUNT will be accessed.
fetch_and_stop "g4dn.xlarge" "fold2000" "oct" "de" "cuda" "cluster-99" "32" "668" "1"
