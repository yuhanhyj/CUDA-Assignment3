#include <cmath>
#include <iostream>
#include <limits>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

using namespace std;

bool         verbose, timing, cpu, gpu;
int          maxIterations;
unsigned int n, numberOfSamples;
double       a, b;

__device__ float exponentialIntegralFloatGPU(const int n, const float x,
                                             int maxIterations) {
  const float eulerConstant = 0.5772156649015329;
  const float epsilon       = 1.E-30;
  const float bigfloat      = 3.402823466e+38F;
  int         i, ii, nm1 = n - 1;
  float       a_, b_, c, d, del, fact, h, psi, ans = 0.0;

  if (n == 0)
    return expf(-x) / x;

  if (x > 1.0f) {
    b_ = x + n;
    c  = bigfloat;
    d  = 1.0f / b_;
    h  = d;
    for (i = 1; i <= maxIterations; i++) {
      a_ = -i * (nm1 + i);
      b_ += 2.0f;
      d   = 1.0f / (a_ * d + b_);
      c   = b_ + a_ / c;
      del = c * d;
      h *= del;
      if (fabsf(del - 1.0f) <= epsilon)
        return h * expf(-x);
    }
    return h * expf(-x);
  } else {
    ans  = (nm1 != 0) ? 1.0f / nm1 : -logf(x) - eulerConstant;
    fact = 1.0f;
    for (i = 1; i <= maxIterations; i++) {
      fact *= -x / i;
      if (i != nm1) {
        del = -fact / (i - nm1);
      } else {
        psi = -eulerConstant;
        for (ii = 1; ii <= nm1; ii++)
          psi += 1.0f / ii;
        del = fact * (-logf(x) + psi);
      }
      ans += del;
      if (fabsf(del) < fabsf(ans) * epsilon)
        return ans;
    }
    return ans;
  }
}

__global__ void computeExponentialIntegral(float* result, int n,
                                           int numberOfSamples, float a,
                                           float b, int maxIterations) {
  int i     = blockIdx.x * blockDim.x + threadIdx.x;
  int total = n * numberOfSamples;
  if (i < total) {
    int   order  = i / numberOfSamples + 1;
    int   sample = i % numberOfSamples + 1;
    float x      = a + sample * ((b - a) / numberOfSamples);
    result[i]    = exponentialIntegralFloatGPU(order, x, maxIterations);
  }
}

float exponentialIntegralFloatCPU(const int n, const float x) {
  const float eulerConstant = 0.5772156649015329;
  const float epsilon       = 1.E-30;
  const float bigfloat      = std::numeric_limits<float>::max();
  int         i, ii, nm1 = n - 1;
  float       a_, b_, c, d, del, fact, h, psi, ans = 0.0;

  if (n == 0)
    return expf(-x) / x;

  if (x > 1.0f) {
    b_ = x + n;
    c  = bigfloat;
    d  = 1.0f / b_;
    h  = d;
    for (i = 1; i <= maxIterations; i++) {
      a_ = -i * (nm1 + i);
      b_ += 2.0f;
      d   = 1.0f / (a_ * d + b_);
      c   = b_ + a_ / c;
      del = c * d;
      h *= del;
      if (fabsf(del - 1.0f) <= epsilon)
        return h * expf(-x);
    }
    return h * expf(-x);
  } else {
    ans  = (nm1 != 0) ? 1.0f / nm1 : -logf(x) - eulerConstant;
    fact = 1.0f;
    for (i = 1; i <= maxIterations; i++) {
      fact *= -x / i;
      if (i != nm1) {
        del = -fact / (i - nm1);
      } else {
        psi = -eulerConstant;
        for (ii = 1; ii <= nm1; ii++)
          psi += 1.0f / ii;
        del = fact * (-logf(x) + psi);
      }
      ans += del;
      if (fabsf(del) < fabsf(ans) * epsilon)
        return ans;
    }
    return ans;
  }
}

int  parseArguments(int argc, char* argv[]);
void printUsage(void);

int main(int argc, char* argv[]) {
  cpu             = true;
  gpu             = true;
  verbose         = false;
  timing          = false;
  n               = 10;
  numberOfSamples = 10;
  a               = 0.0;
  b               = 10.0;
  maxIterations   = 50000;

  parseArguments(argc, argv);
  unsigned int total = n * numberOfSamples;

  std::vector<float> resultsGpuFloat(total);
  std::vector<float> resultsCpuFloat(total);

  double timeCpu = 0.0, timeGpu = 0.0;

  if (cpu) {
    struct timeval start, end;
    if (timing)
      gettimeofday(&start, NULL);
    for (unsigned int i = 1; i <= n; ++i) {
      for (unsigned int j = 1; j <= numberOfSamples; ++j) {
        float x = a + j * ((b - a) / numberOfSamples);
        resultsCpuFloat[(i - 1) * numberOfSamples + (j - 1)] =
            exponentialIntegralFloatCPU(i, x);
      }
    }
    if (timing) {
      gettimeofday(&end, NULL);
      timeCpu =
          (end.tv_sec - start.tv_sec) + 1e-6 * (end.tv_usec - start.tv_usec);
    }
  }

  if (gpu) {
    float* d_results;
    cudaMalloc(&d_results, sizeof(float) * total);
    dim3           blockSize(256);
    dim3           gridSize((total + blockSize.x - 1) / blockSize.x);
    struct timeval start, end;
    if (timing)
      gettimeofday(&start, NULL);
    computeExponentialIntegral<<<gridSize, blockSize>>>(
        d_results, n, numberOfSamples, a, b, maxIterations);
    cudaMemcpy(resultsGpuFloat.data(), d_results, sizeof(float) * total,
               cudaMemcpyDeviceToHost);
    cudaFree(d_results);
    if (timing) {
      gettimeofday(&end, NULL);
      timeGpu =
          (end.tv_sec - start.tv_sec) + 1e-6 * (end.tv_usec - start.tv_usec);
    }
  }

  if (timing) {
    if (cpu)
      printf("CPU time: %f seconds\n", timeCpu);
    if (gpu)
      printf("GPU time: %f seconds\n", timeGpu);
  }

  if (verbose) {
    for (unsigned int i = 1; i <= n; ++i) {
      for (unsigned int j = 1; j <= numberOfSamples; ++j) {
        float x      = a + j * ((b - a) / numberOfSamples);
        float cpuVal = resultsCpuFloat[(i - 1) * numberOfSamples + (j - 1)];
        float gpuVal = resultsGpuFloat[(i - 1) * numberOfSamples + (j - 1)];
        printf("n=%u, x=%.6f, CPU=%.6f, GPU=%.6f, Δ=%.6e\n", i, x, cpuVal,
               gpuVal, fabs(cpuVal - gpuVal));
      }
    }
  }

  return 0;
}

int parseArguments(int argc, char* argv[]) {
  int c;
  while ((c = getopt(argc, argv, "cghn:m:a:b:tv")) != -1) {
    switch (c) {
      case 'c': cpu = false; break;
      case 'g': gpu = false; break;
      case 'h':
        printUsage();
        exit(0);
        break;
      case 'i': maxIterations = atoi(optarg); break;
      case 'n': n = atoi(optarg); break;
      case 'm': numberOfSamples = atoi(optarg); break;
      case 'a': a = atof(optarg); break;
      case 'b': b = atof(optarg); break;
      case 't': timing = true; break;
      case 'v': verbose = true; break;
      default:
        fprintf(stderr, "Invalid option\n");
        printUsage();
        return -1;
    }
  }
  return 0;
}

void printUsage() {
  printf("exponentialIntegral CUDA+CPU comparison\n");
  printf("usage: ./exponentialIntegral [options]\n");
  printf("  -a val  : interval start (default 0.0)\n");
  printf("  -b val  : interval end (default 10.0)\n");
  printf("  -c      : skip CPU test\n");
  printf("  -g      : skip GPU test\n");
  printf("  -n val  : order up to n (default 10)\n");
  printf("  -m val  : number of samples (default 10)\n");
  printf("  -t      : show timing info\n");
  printf("  -v      : verbose output and comparison\n");
}
