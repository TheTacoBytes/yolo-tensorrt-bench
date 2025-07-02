# Makefile for YOLO TRT Inference (standalone build)

# Determine Conda prefix at runtime, fallback to system CUDA
# Use ?= to allow overriding, and check if empty
CUDA_PREFIX ?= $(shell echo $$CONDA_PREFIX)
ifeq ($(CUDA_PREFIX),)
CUDA_PREFIX := /usr/local/cuda
endif

# Paths
CUDA_INC := $(CUDA_PREFIX)/targets/x86_64-linux/include
CUDA_LIB := $(CUDA_PREFIX)/targets/x86_64-linux/lib
TRT_INC  := /usr/include/x86_64-linux-gnu
TRT_LIB  := /usr/lib/x86_64-linux-gnu

# OpenCV flags via pkg-config
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

# Compiler settings
CXX      := g++
CXXFLAGS := -std=c++17 -I$(CUDA_INC) -I$(TRT_INC) $(OPENCV_CFLAGS)
LDFLAGS  := -L$(CUDA_LIB) -L$(TRT_LIB) -lnvinfer -lnvinfer_plugin -lcudart $(OPENCV_LIBS)

# Target executable and source
TARGET := yolo_trt_infer
SRC    := yolo_trt_inference.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)
