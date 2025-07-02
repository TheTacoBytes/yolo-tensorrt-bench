#include <cstring>
#include <cstdio>
#include <cmath>
#include <numeric>
#include <iostream>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>               
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>

#include <opencv2/opencv.hpp>

using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
  }
};

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <engine.trt> <input_video> <output_csv>\n";
    return 1;
  }
  const char* enginePath = argv[1];
  const char* videoPath  = argv[2];
  const char* csvPath    = argv[3];

  // 1) Deserialize engine
  std::ifstream f(enginePath, std::ios::binary|std::ios::ate);
  if (!f) { perror("Engine file"); return 1; }
  size_t size = f.tellg(); f.seekg(0, std::ios::beg);
  std::vector<char> engineData(size);
  f.read(engineData.data(), size);

  Logger logger;
  IRuntime* runtime = createInferRuntime(logger);
  ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
  IExecutionContext* ctx = engine->createExecutionContext();

  // 2) Hard-coded dims for yolo11n-seg.engine
  constexpr int B        = 1;
  constexpr int C_in     = 3, H_in = 640, W_in = 640;
  constexpr int C_out0   = 116, N0 = 8400;
  constexpr int C_out1   = 32, H1 = 160, W1 = 160;

  size_t inBytes   = B*C_in*H_in*W_in * sizeof(float);
  size_t out0Bytes = B*C_out0*N0       * sizeof(float);
  size_t out1Bytes = B*C_out1*H1*W1    * sizeof(float);

  // 3) Allocate GPU buffers
  void* buffers[3];
  cudaMalloc(&buffers[0], inBytes);
  cudaMalloc(&buffers[1], out0Bytes);
  cudaMalloc(&buffers[2], out1Bytes);

  // 4) Open video
  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened()) { std::cerr<<"Cannot open "<<videoPath<<"\n"; return 1; }

  std::vector<float> times;
  cv::Mat frame, rgb, inp(H_in, W_in, CV_32FC3);

  while (cap.read(frame)) {
    // preprocess: BGR->RGB, resize, normalize to [0,1]
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, rgb, {W_in,H_in});
    rgb.convertTo(inp, CV_32FC3, 1/255.0f);

    // upload input
    cudaMemcpy(buffers[0], inp.ptr<float>(), inBytes, cudaMemcpyHostToDevice);

    // inference + timing
    auto t0 = cv::getTickCount();
    ctx->executeV2(buffers);               // synchronous
    cudaDeviceSynchronize();
    auto t1 = cv::getTickCount();

    float ms = (t1 - t0)*1000.0f/cv::getTickFrequency();
    times.push_back(ms);
  }

  // 5) Release GPU
  cudaFree(buffers[0]);
  cudaFree(buffers[1]);
  cudaFree(buffers[2]);


  // 6) Write CSV
  std::ofstream out(csvPath);
  out<<"frame,ms\n";
  for (size_t i=0; i<times.size(); ++i)
    out<< (i+1) << "," << times[i] << "\n";
  float avg = std::accumulate(times.begin(), times.end(), 0.f)
            / float(times.size());
  out<<"avg,"<<avg<<"\n";

  std::cout<<"Done. Avg inference time: "<<avg<<" ms\n";
  return 0;
}
