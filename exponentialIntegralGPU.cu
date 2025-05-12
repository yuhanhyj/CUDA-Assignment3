#include "exponentialIntegralGPU.h"
#include "util_gpu.cuh"

#include <cstdio>

const int row_per_batch = 1024;

template <typename T>
__global__ void exponential_integral_kernel(
    const double a, const double division, const int maxIterations,
    const int blocks_per_col, const int start_row, T* results_gpu) {}

void exponentialIntegralGpu(const unsigned int numberOfSamples,
                            const unsigned int n, const double a,
                            const double division, const int maxIterations,
                            std::vector<std::vector<float>>& resultsGpu,
                            const int block_size, float timings[5]) {

  TIME_INIT();

  TIME_START();
  const int blocks_per_col = div_up(n, block_size);
  const int grid_size      = row_per_batch;
  TIME_END();

  TIME_START();
  float* results_gpu = NULL;
  CHECK_CUDA(
      cudaMalloc((void**) (&results_gpu), n * numberOfSamples * sizeof(float)));
  TIME_END();

  TIME_START();
  CHECK_CUDA(cudaMemset(results_gpu, 0, n * numberOfSamples * sizeof(float)));
  TIME_END();

  TIME_START();
  for (int start_row = 0; start_row < numberOfSamples;
       start_row += row_per_batch) {
    exponential_integral_kernel<<<grid_size, block_size>>>(
        a, division, maxIterations, blocks_per_col, start_row, results_gpu);
  }
  TIME_END();

  TIME_START();
  CHECK_CUDA(cudaMemcpy(resultsGpu.data(), results_gpu,
                        n * numberOfSamples * sizeof(float),
                        cudaMemcpyDeviceToHost));
  TIME_END();

  cudaFree(results_gpu);

  TIME_FINISH();

  return;
}

void exponentialIntegralGpu(const unsigned int numberOfSamples,
                            const unsigned int n, const double a,
                            const double division, const int maxIterations,
                            std::vector<std::vector<double>>& resultsGpu,
                            const int block_size, float timings[5]) {
  TIME_INIT();

  TIME_START();
  const int blocks_per_col = div_up(n, block_size);
  const int grid_size      = row_per_batch;
  TIME_END();

  TIME_START();
  float* results_gpu = NULL;
  CHECK_CUDA(cudaMalloc((void**) (&results_gpu),
                        n * numberOfSamples * sizeof(double)));
  TIME_END();

  TIME_START();
  CHECK_CUDA(cudaMemset(results_gpu, 0, n * numberOfSamples * sizeof(double)));
  TIME_END();

  TIME_START();
  for (int start_row = 0; start_row < numberOfSamples;
       start_row += row_per_batch) {
    exponential_integral_kernel<<<grid_size, block_size>>>(
        a, division, maxIterations, blocks_per_col, start_row, results_gpu);
  }
  TIME_END();

  TIME_START();
  CHECK_CUDA(cudaMemcpy(resultsGpu.data(), results_gpu,
                        n * numberOfSamples * sizeof(double),
                        cudaMemcpyDeviceToHost));
  TIME_END();

  cudaFree(results_gpu);

  TIME_FINISH();

  return;
}
