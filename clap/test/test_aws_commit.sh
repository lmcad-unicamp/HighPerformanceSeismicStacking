#! /bin/bash

HEAD_COMMIT_ID=${1}
COMMIT_BRANCH="master"

FRAMEWORK="cuda"

PROJECT_ROOT="/home/ubuntu/HighPerformanceSeismicStacking"
NFS_MOUNT_POINT="/home/ubuntu/nfs"

DATA_DIR="${NFS_MOUNT_POINT}/data"
BIN_DIR="${PROJECT_ROOT}/${FRAMEWORK}/bin"

CLAPP_SOURCE_PATH="/home/gustavociotto/clap"

GENERATIONS="32"
POPULATION_SIZE="32"

function compile_binary {
    NODE_ID=${1}
    COMPUTE_CAPABILITY=${2}

    if [ "${FRAMEWORK}" == "cuda" ]; then
        ARCH_PARAMETER="ARCH=${COMPUTE_CAPABILITY}"
    fi

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="make all -j16 ${ARCH_PARAMETER}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}")
}

function mkdir_node {
    NODE_ID=${1}
    DIR_TO_BE_CREATED=${2}

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="mkdir -p ${DIR_TO_BE_CREATED}")
}

function setup_node {
    NODE_ID=${1}

    (set -x; \
    clapp group add commands-common ${NODE_ID}; \
    clapp group add ec2-efs ${NODE_ID}; \
    clapp group action ec2-efs mount \
        --nodes ${NODE_ID} \
        --extra efs_mount_ip="172.31.15.69" \
        efs_mount_point="${NFS_MOUNT_POINT}")
}

function start_node {
    AWS_INSTANCE=${1}

    CLAPP_START_OUTPUT=$(clapp node start spits-worker-instance-aws-${AWS_INSTANCE})

    # Node(id=`node-232`, type=`spits-worker-instance-aws-g4dn.xlarge`,
    # status=`reachable`, public_ip=`34.236.216.26`, groups=``, tags=``, last_update=`19-12-20 08:49:44)`
    # Get node id from command's output
    echo $(echo ${CLAPP_START_OUTPUT} | grep -o "Node(id=\`node-[0-9]*\`" | grep -o "node-[0-9]*")
}

function stop_node {
    NODE_ID=${1}

    (set -x; \
    clapp node stop ${NODE_ID})
}

function update_repo {
    NODE_ID=${1}

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="git clean -xdf" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}"; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="git pull origin ${COMMIT_BRANCH}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}";
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="git checkout ${HEAD_COMMIT_ID}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}")
}

function greedy_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_DIR=${3}
    TEST_INDEX=${4}
    NODE_ID=${5}
    THREAD_COUNT=${6}

    TEST_OUTPUT_DIR="${TEST_DIR}/cmp/greedy/${FRAMEWORK}/${DATA_NAME}_${THREAD_COUNT}_${TEST_INDEX}"

    mkdir_node ${NODE_ID} ${TEST_OUTPUT_DIR}

    COMMAND="${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 1024 \
        --traveltime cmp \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log"

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="${COMMAND}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}")
}

function greedy_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_DIR=${3}
    TEST_INDEX=${4}
    NODE_ID=${5}
    THREAD_COUNT=${6}

    TEST_OUTPUT_DIR="${TEST_DIR}/zocrs/greedy/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    mkdir_node ${NODE_ID} ${TEST_OUTPUT_DIR}

    COMMAND="${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --granularity 100 10 10 \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log"

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="${COMMAND}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}")
}

function greedy_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_DIR=${3}
    TEST_INDEX=${4}
    NODE_ID=${5}
    THREAD_COUNT=${6}

    TEST_OUTPUT_DIR="${TEST_DIR}/oct/greedy/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    mkdir_node ${NODE_ID} ${TEST_OUTPUT_DIR}

    COMMAND="${BIN_DIR}/single_host_linear_search \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --granularity 100 100 \
        --traveltime oct \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log"

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="${COMMAND}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}")
}

function de_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_DIR=${3}
    TEST_INDEX=${4}
    NODE_ID=${5}
    THREAD_COUNT=${6}

    TEST_OUTPUT_DIR="${TEST_DIR}/cmp/de/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    mkdir_node ${NODE_ID} ${TEST_OUTPUT_DIR}

    COMMAND="${BIN_DIR}/single_host_de \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime cmp \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log"

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="${COMMAND}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}")
}

function de_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_DIR=${3}
    TEST_INDEX=${4}
    NODE_ID=${5}
    THREAD_COUNT=${6}

    TEST_OUTPUT_DIR="${TEST_DIR}/zocrs/de/${FRAMEWORK}/${DATA_NAME}_${THREAD_COUNT}_${TEST_INDEX}"

    mkdir_node ${NODE_ID} ${TEST_OUTPUT_DIR}

    COMMAND="${BIN_DIR}/single_host_de \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log"

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="${COMMAND}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}")
}

function de_offset_continuation_trajectory {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_DIR=${3}
    TEST_INDEX=${4}
    NODE_ID=${5}
    THREAD_COUNT=${6}

    TEST_OUTPUT_DIR="${TEST_DIR}/oct/de/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    mkdir_node ${NODE_ID} ${TEST_OUTPUT_DIR}

    COMMAND="${BIN_DIR}/single_host_de \
        --aph 1000 --apm 150 --azimuth ${AZIMUTH} --tau 0.02 \
        --input ${DATA_DIR}/${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --generations ${GENERATIONS} --population-size ${POPULATION_SIZE} \
        --traveltime oct \
        --thread-count ${THREAD_COUNT} \
        --verbose 1 | tee ${TEST_OUTPUT_DIR}/output.log"

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="${COMMAND}" \
        workdir="${PROJECT_ROOT}/${FRAMEWORK}")
}


function test {
    NODE_ID=${1}
    COMPUTE_CAPABILITY=${2}
    TEST_DIR=${3}

    setup_node ${NODE_ID}

    update_repo ${NODE_ID}

    compile_binary ${NODE_ID} ${COMPUTE_CAPABILITY}

    # THREAD_COUNT=32

    THREAD_COUNTS=("32" "64" "128" "256" "512" "1024")

    # #### Common Mid Point
    for i in `seq 1 3`; do
        echo "Executing ${i}th iteration - CMP - ${AWS_INSTANCE}"

        for THREAD_COUNT in ${THREAD_COUNTS[@]}; do
            # greedy_common_mid_point "fold2000" "90" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
            greedy_common_mid_point "simple-synthetic" "0" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
        done
    #     de_common_mid_point "fold2000" "90" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    #     greedy_common_mid_point "simple-synthetic" "0" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    #     de_common_mid_point "simple-synthetic" "0" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    done

    #### Zero Offset Common Reflection Point - DE
    # for i in `seq 1 3`;
    # do
    #     echo "Executing ${i}th iteration - ZOCRS - ${AWS_INSTANCE}"
    #     for THREAD_COUNT in ${THREAD_COUNTS[@]}; do
    #         de_zero_offset_reflection_surface "fold2000" "90" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    #     done
    # done

    # #### Offset Continuation Trajectory - DE
    # for i in `seq 1 3`;
    # do
    #     echo "Executing ${i}th iteration - OCT - ${AWS_INSTANCE}"
    #     de_offset_continuation_trajectory "fold2000" "90" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    #     de_offset_continuation_trajectory "simple-synthetic" "0" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    # done

    #### Zero Offset Common Reflection Point - GREEDY
    # for i in `seq 1 3`;
    # do
    #     echo "Executing ${i}th iteration - ZOCRS - ${AWS_INSTANCE}"
    #     greedy_zero_offset_reflection_surface "fold2000" "90" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    #     greedy_zero_offset_reflection_surface "simple-synthetic" "0" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    # done

    # #### Offset Continuation Trajectory - GREEDY
    # for i in `seq 1 3`;
    # do
    #     echo "Executing ${i}th iteration - OCT - ${AWS_INSTANCE}"
    #     greedy_offset_continuation_trajectory "fold2000" "90" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    #     greedy_offset_continuation_trajectory "simple-synthetic" "0" ${TEST_DIR} ${i} ${NODE_ID} ${THREAD_COUNT}
    # done

    stop_node ${NODE_ID}
}

. ${CLAPP_SOURCE_PATH}/clap-env/bin/activate

AWS_INSTANCES=("g4dn.xlarge" "p3.2xlarge" "p2.xlarge")

PIDS=""

for AWS_INSTANCE in ${AWS_INSTANCES[@]}; do
    TEST_ID="${HEAD_COMMIT_ID}_${AWS_INSTANCE}"
    TEST_DIR="${NFS_MOUNT_POINT}/out/${TEST_ID}"

    NODE_ID=$(start_node ${AWS_INSTANCE})

    if [[ ${AWS_INSTANCE} == *"g4dn"* ]]; then
        COMPUTE_CAPABILITY="sm_75"
    elif [[ ${AWS_INSTANCE} == *"p3"* ]]; then
        COMPUTE_CAPABILITY="sm_70"
    elif [[ ${AWS_INSTANCE} == *"p2"* ]]; then
        COMPUTE_CAPABILITY="sm_37"
    fi

    echo $NODE_ID

    test ${NODE_ID} ${COMPUTE_CAPABILITY} ${TEST_DIR} &

    PIDS="${PIDS} $!"
done

echo "Waiting for PIDS=${PIDS}"
wait ${PIDS}
