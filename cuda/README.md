# CUDA

CUDA specific source code to process seismic data.

## Building

A Makefile is provided in order to generate all available binaries. Before starting, assure yourself to set up your environment with the latest CUDA libraries and APIs.
After doing that, locate the CUDA's header location (usually in **/usr/local/cuda/include/**) and edit *CUDA_LIBRARY_PATH* and *NVCC* variables inside this folder's `Makefile` accordingly.
Also check the compute capabilities of your graphic card and use **ARCH** when calling **make**. For example:

```
$ make all -j16 ARCH=75
```

Six differents binaries should be created by the above command into `bin/` folder:

* `single_host_linear_search`: single host implementation for linear (greedy) search.
* `single_host_de`: single host implementation for differential evolution based search.
* `single_host_stretch_free`: to be used after the parameters have been computed. It stackes the data assuring the final result is stretch free.
* `spitz_linear_search`: spits implementation for linear (greedy) search.
* `spitz_de`: spits implementation for differential evolution based search.

`make clean` cleans up your workspace.

## Running

In the following subsections, we present some sample commands that can be used to execute the proposed CUDA implementation. Before going through these sections, it's important to understand what the parameters mean. Refer to the table below for a brief description about them:

| **Parameter**   | **Description**                                                                                                                                          |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| APM             | Modular maximum threshold for a trace's midpoint. A trace is considered if its midpoint `m_0` is `m_0` <= `\|APM\|`.                                     |
| APH             | Modular maximum threshold for a trace's half-offset. A trace is considered if its half-offset `h_0` is `h_0` <= `\|APH\|`.                               |
| AZIMUTH         | Angle in degrees between the coordinate axis for which the measures were made and a reference direction.                                                 |
| TAU             | Parameter used to compute the semblance processing window, which is given by `w = 2 * tau + 1`.                                                          |
| GENERATIONS     | Number of generations. It applies to the **differential evolution** method only.                                                                         |
| POPULATION_SIZE | The number of different values that is going to be considered for each attribute. It also applies to the **differential evolution** compute method only. |
| THREAD_COUNT    | GPU's thread count per block.  |

It's worthy mentioning that `THREAD_COUNT` varies according to the GPU you're running this solution.

### Sample parameter data

This repo provides two synthetic sample data sets, `simple-synthetic` and `fold2000`. Table below summarizes the parameters that can be used for this two files:

| **Data set**       | **APM** | **APH** | AZIMUTH | **TAU** | **POPULATION_SIZE** | **GENERATIONS** |
|--------------------|---------|---------|---------|---------|---------------------|-----------------|
| `simple-synthetic` | 150     | 1000    | 0       | 0.02    | 32                  | 32              |
| `fold2000`         | 150     | 1000    | 90      | 0.02    | 32                  | 32              |

### Single host + differential evolution

To run considering the **Common Mid Point** traveltime model and differential evolution algorithm, use the following command:

```
$ bin/single_host_de \
        --aph ${APH} --apm ${APM} --azimuth ${AZIMUTH} --tau ${TAU} \
        --input ${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} \
        --population-size {POPULATION_SIZE} \
        --traveltime cmp \
        --thread-count ${THREAD_COUNT} \
        --verbose 1
```

For the **Zero Offset Common Reflection Surface** model, use the command below:

```
$ bin/single_host_de \
        --aph ${APH} --apm ${APM} --azimuth ${AZIMUTH} --tau ${TAU} \
        --input ${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 --upper-bounds 6000 \
        --generations ${GENERATIONS} \
        --population-size ${POPULATION_SIZE} \
        --traveltime zocrs --ain 60 --v0 2000 --bpctg 0.1 \
        --thread-count ${THREAD_COUNT} \
        --verbose 1
```

Finally, for the **Offset Continuation Trajectory** traveltime:

```
$ bin/single_host_de \
        --aph ${APH} --apm ${APM} --azimuth ${AZIMUTH} --tau ${TAU} \
        --input ${DATA_NAME}.su \
        --output ${TEST_OUTPUT_DIR} \
        --lower-bounds 450 -0.001 --upper-bounds 6000 0.001 \
        --generations ${GENERATIONS} \
        --population-size ${POPULATION_SIZE} \
        --traveltime oct \
        --thread-count ${THREAD_COUNT} \
        --verbose 1
```

### Test scripts for single node

We provide in [this folder](https://github.com/lmcad-unicamp/HighPerformanceSeismicStacking/tree/master/cuda/test) test scripts that may be used to run each one of the methods above. `test_de.sh` and `test_greedy.sh` have been designed to run the two sample data sets and the differential evolution or greedy methods, respectively. Edit the `TEST_ID` variable and uncomment the lines regarding the execution types you wish to perform. **The scripts don't compile the binaries**, therefore be aware to compile the binaries before executing them.

For example, if you want to run the Offset Continuation Trajectory using the differential evolution compute method, uncomment the following lines:

```
THREAD_COUNT=64
for i in `seq 1 10`;
do
    echo "Executing ${i}th iteration - OCT"
    de_offset_continuation_trajectory "fold2000" "90" ${i} ${THREAD_COUNT}
    de_offset_continuation_trajectory "simple-synthetic" "0" ${i} ${THREAD_COUNT}
done
```

This will run 10 times OCT and DE for the two sample data sets.

### CLAP

For both single and multi-node executions in the cloud, it's suggested to set and use the CLAP tool. Refer to the [official documentation](https://clap.readthedocs.io/en/latest/) for the instructions on how to set it up.

Sample configuration files and test scripst can be found at [`clap/configs`](https://github.com/lmcad-unicamp/HighPerformanceSeismicStacking/tree/master/clap/configs) and [`clap/test`](https://github.com/lmcad-unicamp/HighPerformanceSeismicStacking/tree/master/clap/test) folders, respectively in this repository's root directory.

#### Using CLAP for single node executions

To execute single node executions using CLAP, refer to [this test script](https://github.com/lmcad-unicamp/HighPerformanceSeismicStacking/blob/master/clap/test/test_aws_commit.sh). It receives two input parameters, which are the commit and branch of the source code. We developed this feature in order to allow developers to easily compare two different versions without having to log into all hosts and set their repositories one-by-one.

This script will launch the EC2 hosts (for the job manager and worker), set them up, and execute the program. In the `test` function, remember to uncomment the lines you wish to run. Also, define which instance types you want to launch in by defining the array `AWS_INSTANCES`. For a complete list of supported instance types, refer to [instances.yaml](https://github.com/lmcad-unicamp/HighPerformanceSeismicStacking/blob/master/clap/configs/instances.yaml). If you want to specify the number of threads per block, edit `THREAD_COUNT` accordingly.

It's important to mention that, in case of more than one instance type is specified, the tests are run in parallel.

#### Using CLAP for multi node executions

As the previous section, we've also developed a [test script](https://github.com/lmcad-unicamp/HighPerformanceSeismicStacking/blob/master/clap/test/test_spits.sh) to deploy our application in many nodes. Remember to update `HEAD_COMMIT_ID` with the commit id you want to execute. Similar to the last case, edit `THREAD_COUNT` to change the number of threads per block. Another important parameters to set are:

* `AWS_INSTANCE`: the type of instance you'll be using for the workers.
* `TASK_COUNT` is the number of worker nodes. You also need to edit [your clusters.yml file](https://github.com/lmcad-unicamp/HighPerformanceSeismicStacking/blob/master/clap/configs/clusters/clusters.yml) and change `spits-taskmanager`'s `count` parameter.
* If you want to run the CUDA solution, set `FRAMEWORK` to `cuda`.

Once the execution finishes, each node will save its logs into `SPITS_JOB_PATH`. To fetch these files and save them to EFS mount point, use [fetch_spits_data.sh](https://github.com/lmcad-unicamp/HighPerformanceSeismicStacking/blob/master/clap/test/fetch_spits_data.sh). This script will resume the hosts (if needed) and copy all the files to a common EFS volume, which you can access afterwards. Remember to edit `HEAD_COMMIT_ID` (used only to create a folder containing the this id in its name). Follow the instructions in the script itself:

```
Use the following example to fetch logs and data from all nodes.
fetch_and_stop AWS_INSTANCE DATA_NAME TRAVELTIME COMPUTE_ALGORITHM
FRAMEWORK CLUSTER_ID NODE_COUNT STARTING_NODE TEST_INDEX

* AWS_INSTANCE, DATA_NAME, TRAVELTIME, COMPUTE_ALGORITHM, FRAMEWORK, and TEST_INDEX are meant to identify the experiment.
* CLUSTER_ID, NODE_COUNT and STARTING_NODE are meant to identify cluster and nodes. All nodes from STARTING_NODE to STARTING_NODE + NODE_COUNT will be accessed.
```