#! /bin/bash

HEAD_COMMIT_ID="0447d1c"

CLAPP_SOURCE_PATH="/home/gustavociotto/clap"
CLAPP_CLUSTER="clapp cluster"

NFS_MOUNT_POINT="/home/ubuntu/nfs"

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

function fetch_and_stop {
    AWS_INSTANCE=${1}
    DATA_NAME=${2}
    TRAVELTIME=${3}
    COMPUTE_ALGORITHM=${4}
    FRAMEWORK=${5}
    CLUSTER_ID=${6}
    NODE_COUNT=${7}
    STARTING_NODE=${8}

    TEST_ID="${HEAD_COMMIT_ID}_spits_${AWS_INSTANCE}"
    TEST_DIR="${NFS_MOUNT_POINT}/out/${TEST_ID}/${TRAVELTIME}/${COMPUTE_ALGORITHM}/${FRAMEWORK}/${DATA_NAME}/${NODE_COUNT}-node"

    JOB_MANAGER_NODE_ID="node-${STARTING_NODE}"
    JOB_MANAGER_OUTPUT_DIR="${TEST_DIR}/jobmanager"

    TASK_MANAGER_OUTPUT_DIR="${TEST_DIR}/taskmanager"

    resume_cluster ${CLUSTER_ID}

    mkdir_node ${JOB_MANAGER_NODE_ID} ${JOB_MANAGER_OUTPUT_DIR}
    copy_node ${JOB_MANAGER_NODE_ID} ${SPITS_JOB_PATH} ${JOB_MANAGER_OUTPUT_DIR}

    for TASK_MANAGER_NODE_IDX in `seq 1 ${NODE_COUNT}`; do
        TASK_MANAGER_NODE_ID="node-$(expr $STARTING_NODE + $TASK_MANAGER_NODE_IDX)"

        TASK_MANAGER_NODE_OUT_DIR="${TASK_MANAGER_OUTPUT_DIR}/${TASK_MANAGER_NODE_IDX}"

        mkdir_node ${TASK_MANAGER_NODE_ID} ${TASK_MANAGER_NODE_OUT_DIR}
        copy_node ${TASK_MANAGER_NODE_ID} ${SPITS_JOB_PATH} ${TASK_MANAGER_NODE_OUT_DIR}
    done

    stop_cluster ${CLUSTER_ID}

}

. ${CLAPP_SOURCE_PATH}/clap-env/bin/activate

# fetch_and_stop "p2.xlarge" "simple-synthetic" "oct" "de" "cuda" "cluster-19" "4" "93"
# fetch_and_stop "p2.xlarge" "simple-synthetic" "oct" "de" "cuda" "cluster-20" "8" "98"

# fetch_and_stop "p2.xlarge" "fold2000" "oct" "de" "cuda" "cluster-21" "1" "107"
# fetch_and_stop "p2.xlarge" "fold2000" "oct" "de" "cuda" "cluster-22" "2" "109"
# fetch_and_stop "p2.xlarge" "fold2000" "oct" "de" "cuda" "cluster-23" "4" "112"
# fetch_and_stop "p2.xlarge" "fold2000" "oct" "de" "cuda" "cluster-24" "8" "117"
# fetch_and_stop "p2.xlarge" "fold2000" "oct" "de" "cuda" "cluster-25" "8" "126"

# fetch_and_stop "g4dn.xlarge" "simple-synthetic" "zocrs" "de" "cuda" "cluster-26" "1" "135"
# fetch_and_stop "g4dn.xlarge" "simple-synthetic" "zocrs" "de" "cuda" "cluster-27" "2" "137"
# fetch_and_stop "g4dn.xlarge" "simple-synthetic" "zocrs" "de" "cuda" "cluster-28" "4" "140"
# fetch_and_stop "g4dn.xlarge" "simple-synthetic" "zocrs" "de" "cuda" "cluster-29" "8" "145"

# fetch_and_stop "g4dn.xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-30" "1" "154"
# fetch_and_stop "g4dn.xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-31" "2" "156"
# fetch_and_stop "g4dn.xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-32" "4" "159"
# fetch_and_stop "g4dn.xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-33" "8" "166"

# fetch_and_stop "g4dn.xlarge" "simple-synthetic" "oct" "de" "cuda" "cluster-34" "1" "177"
# fetch_and_stop "g4dn.xlarge" "simple-synthetic" "oct" "de" "cuda" "cluster-35" "2" "181"
# fetch_and_stop "g4dn.xlarge" "simple-synthetic" "oct" "de" "cuda" "cluster-36" "4" "186"
# fetch_and_stop "g4dn.xlarge" "simple-synthetic" "oct" "de" "cuda" "cluster-37" "8" "191"

# fetch_and_stop "g4dn.xlarge" "fold2000" "oct" "de" "cuda" "cluster-39" "1" "201"
# fetch_and_stop "g4dn.xlarge" "fold2000" "oct" "de" "cuda" "cluster-40" "2" "203"
# fetch_and_stop "g4dn.xlarge" "fold2000" "oct" "de" "cuda" "cluster-41" "4" "206"
# fetch_and_stop "g4dn.xlarge" "fold2000" "oct" "de" "cuda" "cluster-42" "8" "211"

# fetch_and_stop "g4dn.12xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-46" "1" "226"
# fetch_and_stop "g4dn.12xlarge" "fold2000" "oct" "de" "cuda" "cluster-43" "1" "220"

# fetch_and_stop "p2.8xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-45" "1" "224"
# fetch_and_stop "p2.8xlarge" "fold2000" "oct" "de" "cuda" "cluster-44" "1" "222"

# fetch_and_stop "g4dn.12xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-48" "1" "247"
fetch_and_stop "p2.8xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-49" "1" "249"

fetch_and_stop "g4dn.xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-50" "4" "251"

fetch_and_stop "p2.xlarge" "fold2000" "zocrs" "de" "cuda" "cluster-51" "8" "256"
