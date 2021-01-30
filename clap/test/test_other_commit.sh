#! /bin/bash

HEAD_COMMIT_ID=${1}

FRAMEWORK="cuda"

GIT_REPO="https://github.com/hpg-cepetro/IPDPS-CRS-CMP-code.git"
PROJECT_ROOT="/home/ubuntu/IPDPS-CRS-CMP-code"
NFS_MOUNT_POINT="/home/ubuntu/nfs"

DATA_DIR="${NFS_MOUNT_POINT}/data"
BIN_DIR="${PROJECT_ROOT}/${FRAMEWORK}/bin"

CLAPP_SOURCE_PATH="/home/gustavociotto/clap"

GENERATIONS="32"
POPULATION_SIZE="32"

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
        --extra cmd="git clone ${GIT_REPO}" \
        workdir="/home/ubuntu"; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="git checkout ${HEAD_COMMIT_ID}" \
        workdir="${PROJECT_ROOT}")
}

function greedy_common_mid_point {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_DIR=${3}
    TEST_INDEX=${4}
    NODE_ID=${5}

    TEST_OUTPUT_DIR="${TEST_DIR}/cmp/greedy/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    BIN_DIR="${PROJECT_ROOT}/CMP/CUDA/bin"

    mkdir_node ${NODE_ID} ${BIN_DIR}

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="/usr/local/cuda/bin/nvcc -O3 -g -DNDEBUG -std=c++11 --use_fast_math \
                          -gencode=arch=compute_75,code=sm_75 \
                          -gencode=arch=compute_70,code=sm_70 \
                          src/*.cpp src/*.cu -I./include/ -lcuda -lm -o bin/cmp-cuda" \
        workdir="${PROJECT_ROOT}/CMP/CUDA/")

    mkdir_node ${NODE_ID} ${TEST_OUTPUT_DIR}

    COMMAND="./cmp-cuda \
        -aph 1000 -tau 0.02 \
        -i ${DATA_DIR}/${DATA_NAME}.su \
        -c0 1.11e-7 -c1 1.97e-5  \
        -nc 1024 \
        -v 3 | tee ${TEST_OUTPUT_DIR}/output.log"

    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="${COMMAND}" \
        workdir="${BIN_DIR}"
}

function greedy_zero_offset_reflection_surface {
    DATA_NAME=${1}
    AZIMUTH=${2}
    TEST_DIR=${3}
    TEST_INDEX=${4}
    NODE_ID=${5}
    COMPUTE_CAPABILITY=${6}

    TEST_OUTPUT_DIR="${TEST_DIR}/zocrs/greedy/${FRAMEWORK}/${DATA_NAME}_${TEST_INDEX}"

    BIN_DIR="${PROJECT_ROOT}/CRS/CUDA/bin"

    mkdir_node ${NODE_ID} ${BIN_DIR}

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="/usr/local/cuda/bin/nvcc -O3 -g -DNDEBUG -std=c++11 --use_fast_math \
                          -gencode=arch=compute_75,code=sm_75 \
                          -gencode=arch=compute_70,code=sm_70 \
                          src/*.cpp src/*.cu -I./include/ -lcuda -lm -o bin/crs-cuda" \
        workdir="${PROJECT_ROOT}/CRS/CUDA/")

    mkdir_node ${NODE_ID} ${TEST_OUTPUT_DIR}

    COMMAND="./crs-cuda \
        -aph 1000 -apm 150 -tau 0.02 \
        -i ${DATA_DIR}/${DATA_NAME}.su \
        -na 10 -nb 10 -nc 100 \
        -a0 -0.7e-3 -a1 0.7e-3 -b0 -1e-7 -b1 1e-7 -c0 1.11e-7 -c1 1.97e-5 \
        -v 3 | tee ${TEST_OUTPUT_DIR}/output.log"

    (set -x; \
    clapp group action commands-common run-command \
        --nodes ${NODE_ID} \
        --extra cmd="${COMMAND}" \
        workdir="${BIN_DIR}")
}

function test {
    NODE_ID=${1}
    COMPUTE_CAPABILITY=${2}
    TEST_DIR=${3}

    setup_node ${NODE_ID}

    update_repo ${NODE_ID}

    #### Common Mid Point
    for i in `seq 1 3`;
    do
        echo "Executing ${i}th iteration - CMP - ${AWS_INSTANCE}"
        greedy_common_mid_point "fold2000" "90" ${TEST_DIR} ${i} ${NODE_ID}
        greedy_common_mid_point "simple-synthetic" "0" ${TEST_DIR} ${i} ${NODE_ID}
    done

    #### Zero Offset Common Reflection Point - GREEDY
    for i in `seq 1 3`;
    do
        echo "Executing ${i}th iteration - ZOCRS - ${AWS_INSTANCE}"
        greedy_zero_offset_reflection_surface "fold2000" "90" ${TEST_DIR} ${i} ${NODE_ID}
        greedy_zero_offset_reflection_surface "simple-synthetic" "0" ${TEST_DIR} ${i} ${NODE_ID}
    done

    stop_node ${NODE_ID}
}

. ${CLAPP_SOURCE_PATH}/clap-env/bin/activate

AWS_INSTANCES=("g4dn.xlarge" "p3.2xlarge")

PIDS=""

for AWS_INSTANCE in ${AWS_INSTANCES[@]}; do
    TEST_ID="${HEAD_COMMIT_ID}_${AWS_INSTANCE}_other"
    TEST_DIR="${NFS_MOUNT_POINT}/out/${TEST_ID}"

    NODE_ID=$(start_node ${AWS_INSTANCE})

    echo $NODE_ID

    if [[ ${AWS_INSTANCE} == *"g4dn"* ]]; then
        COMPUTE_CAPABILITY="sm_75"
    elif [[ ${AWS_INSTANCE} == *"p3"* ]]; then
        COMPUTE_CAPABILITY="sm_70"
    fi

    test ${NODE_ID} ${COMPUTE_CAPABILITY} ${TEST_DIR} &

    PIDS="${PIDS} $!"
done

echo "Waiting for PIDS=${PIDS}"
wait ${PIDS}
