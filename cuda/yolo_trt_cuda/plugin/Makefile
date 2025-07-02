# plugin/Makefile — build YoloNMSPlugin + yolo_trt_infer demo

# 1) System CUDA 12.8 install
CUDA_ROOT := /usr/local/cuda-12.8
CUDA_INC  := -I$(CUDA_ROOT)/targets/x86_64-linux/include
CUDA_LIB  := -L$(CUDA_ROOT)/targets/x86_64-linux/lib
NVCC      := nvcc

# 2) TensorRT paths
TRT_INC    := -I/usr/include/x86_64-linux-gnu
TRT_LIB    := -L/usr/lib/x86_64-linux-gnu -lnvinfer -lnvinfer_plugin

# 3) OpenCV via pkg-config
OPENCV_CFLAGS := $(shell pkg-config --cflags  opencv4)
OPENCV_LIBS   := $(shell pkg-config --libs    opencv4)

# 4) compilers
CXX        := g++
NVCC_FLAGS := -std=c++17 -Xcompiler -fPIC $(CUDA_INC) $(TRT_INC) $(OPENCV_CFLAGS)
CXX_FLAGS  := -std=c++17 -fPIC   $(CUDA_INC) $(TRT_INC) $(OPENCV_CFLAGS)

PLUGIN_LDFLAGS := -shared $(CUDA_LIB) -lcudart $(TRT_LIB) $(OPENCV_LIBS)
RUN_LDFLAGS    := -L. -lyolonms $(CUDA_LIB) -lcudart $(TRT_LIB) $(OPENCV_LIBS)

.PHONY: all clean
all: libyolonms.so yolo_trt_infer

# 6) Compile your NMS kernel
yolo_nms_kernel.o: yolo_nms_kernel.cu yolo_nms_kernel.h
	$(NVCC) $(NVCC_FLAGS) -ccbin $(CXX) -c $< -o $@

# 7) Compile the plugin C++ parts
YoloNMSPlugin.o:           YoloNMSPlugin.cpp YoloNMSPlugin.h yolo_nms_kernel.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

YoloNMSPluginCreator.o:    YoloNMSPluginCreator.cpp YoloNMSPluginCreator.h
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# 8) Link the shared plugin library
libyolonms.so: yolo_nms_kernel.o YoloNMSPlugin.o YoloNMSPluginCreator.o
	$(CXX) -o $@ $^ $(PLUGIN_LDFLAGS)

# 9) Build the demo, linking in TRT, CUDA, OpenCV _and_ our plugin
yolo_trt_infer: yolo_trt_inference.cpp libyolonms.so
	$(CXX) $(CXX_FLAGS) $< -o $@ $(RUN_LDFLAGS)

clean:
	rm -f *.o libyolonms.so yolo_trt_infer
