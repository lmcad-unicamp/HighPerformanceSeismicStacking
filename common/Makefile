SHELL = /bin/bash

#
# Compiler flags
#

PROJECT_ROOT_PATH = $(dir $(shell pwd))

ENABLE_DEBUG := no
ifeq ($(ENABLE_DEBUG), yes)
	CFLAGS += -g -O0
else
	CFLAGS += -O3
endif

CFLAGS += -Wall -fPIC -std=c++17 -Wno-sign-conversion

LDFLAGS += -lboost_program_options -lboost_system -lboost_filesystem -lm -L/usr/lib/gcc/x86_64-linux-gnu/8/ -lstdc++fs

#
# Folder paths
#

BINARY_PATH = $(PROJECT_ROOT_PATH)/common/bin/
SPITS_PATH = $(PROJECT_ROOT_PATH)/common/main/spitz

CPP_SOURCE = $(shell find $(PROJECT_ROOT_PATH)/common/src -name '*.cpp')
CPP_OBJECT = $(CPP_SOURCE:.cpp=.o)

SPITS_LIBRARY_PATH = $(PROJECT_ROOT_PATH)/spits/API/include/

GPP = g++-8

%.o: %.cpp
	$(GPP) -o $@ -c $< -I$(PROJECT_ROOT_PATH) -I$(SPITS_LIBRARY_PATH) $(CFLAGS)

dependencies: $(CPP_OBJECT)

spitz_linear_search: $(SPITS_PATH)/linear_search.cpp $(CPP_OBJECT)
	$(GPP) -o $(BINARY_PATH)/$@ $^ -shared $(CFLAGS) -I$(PROJECT_ROOT_PATH) -I$(SPITS_LIBRARY_PATH) $(LDFLAGS)

spitz_de: $(SPITS_PATH)/differential_evolution.cpp $(CPP_OBJECT)
	$(GPP) -o $(BINARY_PATH)/$@ $^ -shared $(CFLAGS) -I$(PROJECT_ROOT_PATH) -I$(SPITS_LIBRARY_PATH) $(LDFLAGS)

all: spitz_linear_search spitz_de

clean:
	rm -rf $(CPP_OBJECT)
	rm -rf $(BINARY_PATH)/*
