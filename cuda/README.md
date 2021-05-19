# CUDA

CUDA specific source code to process seismic data.

## Building

A Makefile is provided in order to generate all available binaries. Before starting, assure yourself to set up your environment with the latest CUDA libraries and APIs.
After doing that, locate the CUDA's header location (usually in **/usr/local/cuda/include/**) and edit *CUDA_LIBRARY_PATH* and *NVCC* variables inside this folder's `Makefile` accordingly.
Also check the compute capabilities of your graphic card and use **ARCH=30** when calling **make**. For example:

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

### Single host + differential evolution

To run considering **Common Mid Point** traveltime model and differential evolution algorithm, use the following command:

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

### SPITS + CLAP + differential evolution

For multi-node executions, it's suggested to set and use the CLAP tool.

Sample configuration files and test script can be found at `clap/configs` and `clap/test/test_spits.sh` in this repository's root directory.