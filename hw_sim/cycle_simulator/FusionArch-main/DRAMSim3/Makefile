# ONLY use this makefile if you do NOT have a cmake 3.0+ version

CC=gcc
CXX=g++

FMT_LIB_DIR=ext/fmt/include
INI_LIB_DIR=ext/headers
JSON_LIB_DIR=ext/headers
ARGS_LIB_DIR=ext/headers

PYTHON_INC=/home/liuxueyuan/anaconda3/envs/pointnet2/include/python3.9
PYTHON_LIB_DIR=/home/liuxueyuan/anaconda3/envs/pointnet2/lib/

INC=-Isrc/ -I$(FMT_LIB_DIR) -I$(INI_LIB_DIR) -I$(ARGS_LIB_DIR) -I$(JSON_LIB_DIR) -I$(PYTHON_INC)
# CXXFLAGS=-Wall -O3 -fPIC -std=c++11 $(INC) -DFMT_HEADER_ONLY=1
CXXFLAGS=-Wall -g -fPIC -std=c++11 $(INC) -DFMT_HEADER_ONLY=1
LIB_NAME=libdramsim3.so
EXE_NAME=dramsim3main.out

SRCS = src/bankstate.cc src/channel_state.cc src/command_queue.cc src/common.cc \
		src/configuration.cc src/controller.cc src/dram_system.cc src/hmc.cc \
		src/memory_system.cc src/refresh.cc src/simple_stats.cc src/timing.cc \
		dramsim3_interface.cc

EXE_SRCS = src/cpu.cc src/main.cc
# EXE_SRCS = dramsim3_interface.cc

OBJECTS = $(addsuffix .o, $(basename $(SRCS)))
EXE_OBJS = $(addsuffix .o, $(basename $(EXE_SRCS)))
EXE_OBJS := $(EXE_OBJS) $(OBJECTS)


all: $(LIB_NAME) $(EXE_NAME)

$(EXE_NAME): $(EXE_OBJS)
	$(CXX) $(CXXFLAGS) -g -o $@ $^ -L$(PYTHON_LIB_DIR) -lpython3.9

$(LIB_NAME): $(OBJECTS)
	$(CXX) -g -shared -Wl,-soname,$@ -o $@ $^

%.o : %.cc
	$(CXX)  $(CXXFLAGS) -o $@ -c $<

%.o : %.c
	$(CC) -fPIC -O2 -o $@ -c $<

clean:
	-rm -f $(EXE_OBJS) $(LIB_NAME) $(EXE_NAME)