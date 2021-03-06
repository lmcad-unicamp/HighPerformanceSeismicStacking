SHELL = /bin/bash

#
# Compiler flags
#

PROJECT_ROOT_PATH = $(dir $(shell pwd))

ENABLE_DEBUG := no
ifeq ($(ENABLE_DEBUG), yes)
	CFLAGS += -g -O0
	NVCCFLAGS += -G -g -O0
else
	CFLAGS += -O3
	NVCCFLAGS += -O3
endif

ENABLE_PROFILING := no
ifeq ($(ENABLE_PROFILING), yes)
	CFLAGS += -DPROFILE_ENABLED
	NVCCFLAGS += -DPROFILE_ENABLED
endif

LDFLAGS += -lboost_program_options -lboost_system -lboost_filesystem -lm -lcuda -lcurand -L/usr/lib/gcc/x86_64-linux-gnu/8/ -lstdc++fs
CFLAGS += -Wall -fPIC -std=c++17 -Wno-sign-conversion
NVCCFLAGS += --use_fast_math -Xcompiler -Wall -Xcompiler -fPIC -gencode arch=compute_$(ARCH),code=sm_$(ARCH)

#
# Folder paths
#

BINARY_PATH = $(PROJECT_ROOT_PATH)/cuda/bin/
SINGLE_HOST_PATH = $(PROJECT_ROOT_PATH)/cuda/main/single_host
SPITS_PATH = $(PROJECT_ROOT_PATH)/cuda/main/spitz

CPP_SOURCE = $(shell find $(PROJECT_ROOT_PATH)/{common,cuda}/src -name '*.cpp')
CPP_OBJECT = $(CPP_SOURCE:.cpp=.o)

CUDA_SOURCE = $(shell find $(PROJECT_ROOT_PATH)/cuda/src -name '*.cu')
CUDA_OBJECT = $(CUDA_SOURCE:.cu=.o)

CUDA_LIBRARY_PATH = /usr/local/cuda/include/

SPITS_LIBRARY_PATH = $(PROJECT_ROOT_PATH)/spits/API/include/

NVCC = /usr/local/cuda/bin/nvcc
GPP = g++-8

%.o: %.cu
	$(NVCC) -o $@ -c $< -dc -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I$(SPITS_LIBRARY_PATH) $(NVCCFLAGS)

%.o: %.cpp
	$(GPP) -o $@ -c $< -I$(PROJECT_ROOT_PATH) -I$(SPITS_LIBRARY_PATH) $(CFLAGS)

dependencies: $(CPP_OBJECT) $(CUDA_OBJECT)

single_host_linear_search: $(SINGLE_HOST_PATH)/linear_search.cpp $(CPP_OBJECT) $(CUDA_OBJECT)
	$(NVCC) -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS) $(NVCCFLAGS)

single_host_de: $(SINGLE_HOST_PATH)/differential_evolution.cpp $(CPP_OBJECT) $(CUDA_OBJECT)
	$(NVCC) -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS) $(NVCCFLAGS)

single_host_stretch_free: $(SINGLE_HOST_PATH)/stretch_free.cpp $(CPP_OBJECT) $(CUDA_OBJECT)
	$(NVCC) -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS) $(NVCCFLAGS)

spitz_linear_search: $(SPITS_PATH)/linear_search.cpp $(CPP_OBJECT) $(CUDA_OBJECT)
	$(NVCC) -o $(BINARY_PATH)/$@ $^ -shared -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS) $(NVCCFLAGS)

spitz_de: $(SPITS_PATH)/differential_evolution.cpp $(CPP_OBJECT) $(CUDA_OBJECT)
	$(NVCC) -o $(BINARY_PATH)/$@ $^ -shared -I$(PROJECT_ROOT_PATH) -I$(CUDA_LIBRARY_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS) $(NVCCFLAGS)

all: single_host_linear_search single_host_de single_host_stretch_free spitz_linear_search spitz_de

spits: spitz_linear_search spitz_de

clean:
	rm -rf $(CPP_OBJECT)
	rm -rf $(CUDA_OBJECT)
	rm -rf $(BINARY_PATH)/*
