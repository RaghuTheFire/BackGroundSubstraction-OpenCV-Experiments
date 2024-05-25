#include <cstddef>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/bgsegm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
                                                              
const int iterations = 10;
const std::string results_filename = "../results/results_cpp.csv";
                                                           
double benchmark_cpu_mog2(std::string video_file) 
{
  int frame_counter = 1;
  cv::Mat frame, fgMask;
  cv::Ptr < cv::BackgroundSubtractor > background_subtractor_cpu;
  background_subtractor_cpu = cv::createBackgroundSubtractorMOG2(120, 250., true);
  auto start_time = std::chrono::high_resolution_clock::now();

  cv::VideoCapture capture(video_file);
  if (!capture.isOpened()) 
  {
    std::cerr << "Error opening video" << std::endl;
    return -1.;
  }
  while (true) 
  {
    capture >> frame;
    if (frame.empty()) 
    {
      break;
    }
    background_subtractor_cpu -> apply(frame, fgMask);
    frame_counter += 1;
  }
  capture.release();
  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast <std::chrono::nanoseconds > (end_time - start_time).count();
  elapsed /= 1000000000;
  double fps = frame_counter / elapsed;
  return fps;
}

double benchmark_gpu_mog2(std::string video_file) 
{
  int frame_counter = 1;
  cv::Mat frame;
  cv::cuda::GpuMat gframe, fgMask;
  cv::Ptr < cv::BackgroundSubtractor > background_subtractor_gpu;
  background_subtractor_gpu = cv::cuda::createBackgroundSubtractorMOG2(120, 250., true);
  auto start_time = std::chrono::high_resolution_clock::now();

  cv::VideoCapture capture(video_file);
  if (!capture.isOpened()) 
  {
    std::cerr << "Error opening video" << std::endl;
    return -1.;
  }
  while (true) 
  {
    capture >> frame;
    if (frame.empty()) 
    {
      break;
    }
    gframe.upload(frame);
    background_subtractor_gpu -> apply(gframe, fgMask);
    frame_counter += 1;
  }
  capture.release();
  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast <
    std::chrono::nanoseconds > (end_time - start_time).count();
  elapsed /= 1000000000;
  double fps = frame_counter / elapsed;
  return fps;
}

int main(int argc, char ** argv) 
{
  std::cout << "===================================================" << "=========" << std::endl;
  std::cout << "Running OpenCV C++ vs CUDA Background Subtraction" << " Benchmarks" << std::endl;
  std::cout << "===================================================" << "=========" << std::endl;
  std::ofstream output_file(results_filename);
  if (!output_file) 
  {
    std::cout << "Can't open log file. Aborting benchmark!" << std::endl;
    return 1;
  }
  if (argc < 2) 
  {
    std::cout << "Input video file is missing" << std::endl;
  } 
  else 
  {
    std::string video_file = std::string(argv[1]);
    for (int i = 0; i < iterations; i++) 
    {
      // CPU part
      double fps = benchmark_cpu_mog2(video_file);
      std::string result = video_file + ", MOG2, MOG2 (CPU C++), CPU, C++," + std::to_string(fps) + "\n";
      output_file << result;
      std::cout << result << std::endl;

      // CUDA part
      fps = benchmark_gpu_mog2(video_file);
      result = std::string(argv[1]) + ", MOG2, MOG2 (CUDA C++), CUDA, C++," + std::to_string(fps) + "\n";
      output_file << result;
      std::cout << result << std::endl;
    }
    output_file.close();
  }
  std::cout << "===================================================" << std::endl;
  std::cout << "Benchmark finished" << std::endl;
  std::cout << "===================================================" << std::endl;
  return 0;
}
