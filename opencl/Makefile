SHELL = /bin/bash

#
# Compiler flags
#

ENABLE_DEBUG := no
ifeq ($(ENABLE_DEBUG), yes)
	CFLAGS += -g -O0
else
	CFLAGS += -O3
endif

CFLAGS += -Wall -fPIC -std=c++17 -Wno-sign-conversion

LDFLAGS += -lboost_program_options -lboost_system -lboost_filesystem -lstdc++fs -lm -lOpenCL -pthread

#
# Folder paths
#

PROJECT_ROOT_PATH = ..
BINARY_PATH = $(PROJECT_ROOT_PATH)/opencl/bin/
SINGLE_HOST_PATH = $(PROJECT_ROOT_PATH)/opencl/main/single_host
SPITS_PATH = $(PROJECT_ROOT_PATH)/opencl/main/spitz

CPP_SOURCE = $(shell find $(PROJECT_ROOT_PATH)/{common,opencl}/src -name '*.cpp')
CPP_OBJECT = $(CPP_SOURCE:.cpp=.o)

SPITS_LIBRARY_PATH = $(PROJECT_ROOT_PATH)/spits/API/include/

%.o: %.cpp
	g++ -o $@ -c $< -I$(PROJECT_ROOT_PATH) -I$(SPITS_LIBRARY_PATH) $(CFLAGS)

dependencies: $(CPP_OBJECT)

single_host_linear_search: $(SINGLE_HOST_PATH)/linear_search.cpp $(CPP_OBJECT)
	g++ -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I$(SPITS_LIBRARY_PATH) $(LDFLAGS)

single_host_de: $(SINGLE_HOST_PATH)/differential_evolution.cpp $(CPP_OBJECT)
	g++ -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS)

single_host_stretch_free: $(SINGLE_HOST_PATH)/stretch_free.cpp $(CPP_OBJECT)
	g++ -o $(BINARY_PATH)/$@ $^ -I$(PROJECT_ROOT_PATH) -I${SPITS_LIBRARY_PATH} $(LDFLAGS)

all: single_host_linear_search

clean:
	rm -rf $(CPP_OBJECT)
	rm -rf $(BINARY_PATH)/*
