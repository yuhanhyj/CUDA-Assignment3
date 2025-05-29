#include "exponentialIntegralGPU.h"
#include "util_gpu.cuh"

#include <cstdio>

const int row_per_batch = 512;

template <typename T>
__device__ __forceinline__ T exponentialIntegral(const int n, const T x,
                                                 const int maxIterations) {
  double eulerConstant = 0.5772156649015329;
  double epsilon       = 1.E-30;
  double bigDouble     = 1.0e100;
  int    i, ii, nm1 = n - 1;
  T      a, b, c, d, del, fact, h, psi, ans = 0.0;

  if (n < 0 || x < 0 || (x == 0.0 && ((n == 0) || (n == 1)))) {
    return NAN;
  }

  if (x > 1.0) {
    b = x + n;
    c = bigDouble;
    d = 1.0 / b;
    h = d;
    for (i = 1; i <= maxIterations; i++) {
      a = -i * (nm1 + i);
      b += 2.0;
      d   = 1.0 / (a * d + b);
      c   = b + a / c;
      del = c * d;
      h *= del;
      if (fabs(del - 1.0) <= epsilon) {
        ans = h * exp(-x);
        return ans;
      }
    }
    ans = h * exp(-x);
    return ans;
  } else {
    ans  = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
    fact = 1.0;
    for (i = 1; i <= maxIterations; i++) {
      fact *= -x / i;
      if (i != nm1) {
        del = -fact / (i - nm1);
      } else {
        psi = -eulerConstant;
        for (ii = 1; ii <= nm1; ii++) {
          psi += 1.0 / ii;
        }
        del = fact * (-log(x) + psi);
      }
      ans += del;
      if (fabs(del) < fabs(ans) * epsilon)
        return ans;
    }
    return ans;
  }

  return ans;
}

template <typename T>
__device__ T exponentialIntegralDeekSeek(const int n, const T x,
                                         const int maxIterations) {
  const T eulerConstant = static_cast<T>(0.5772156649015329);
  const T epsilon       = static_cast<T>(1.E-30);
  const T bigValue      = static_cast<T>(1.0E+300);
  int     i, ii, nm1 = n - 1;
  T       a, b, c, d, del, fact, h, psi, ans = static_cast<T>(0.0);

  if (n < 0 || x < 0 || (x == static_cast<T>(0.0) && ((n == 0) || (n == 1)))) {
    printf("Bad arguments passed to exponentialIntegral\n");
    return static_cast<T>(-1.0);
  }

  if (n == 0) {
    ans = exp(-x) / x;
  } else {
    if (x > static_cast<T>(1.0)) {
      // Continued fraction method (for x > 1)
      b = x + n;
      c = bigValue;
      d = static_cast<T>(1.0) / b;
      h = d;
      for (i = 1; i <= maxIterations; i++) {
        a = -i * (nm1 + i);
        b += static_cast<T>(2.0);
        d   = static_cast<T>(1.0) / (a * d + b);
        c   = b + a / c;
        del = c * d;
        h *= del;
        if (fabs(del - static_cast<T>(1.0)) <= epsilon) {
          ans = h * exp(-x);
          return ans;
        }
      }
      ans = h * exp(-x);
      return ans;
    } else {
      // Series expansion method (for x <= 1)
      ans  = (nm1 != 0 ? static_cast<T>(1.0) / nm1 : -log(x) - eulerConstant);
      fact = static_cast<T>(1.0);
      for (i = 1; i <= maxIterations; i++) {
        fact *= -x / i;
        if (i != nm1) {
          del = -fact / (i - nm1);
        } else {
          psi = -eulerConstant;
          for (ii = 1; ii <= nm1; ii++) {
            psi += static_cast<T>(1.0) / ii;
          }
          del = fact * (-log(x) + psi);
        }
        ans += del;
        if (fabs(del) < fabs(ans) * epsilon)
          return ans;
      }
    }
  }
  return ans;
}

// template <typename T>
// __device__ T exponentialIntegralChatGPT(const int n, const T x, const int
// maxIterations = 100) { // have problem in line 125
//     const T eulerConstant = static_cast<T>(0.5772156649015329);
//     const T epsilon = static_cast<T>(1.E-30);
//     const T big = std::numeric_limits<T>::max();

//     int i, ii, nm1 = n - 1;
//     T a, b, c, d, del, fact, h, psi, ans = static_cast<T>(0);

//     if (n < 0 || x < static_cast<T>(0) || (x == static_cast<T>(0) && (n == 0
//     || n == 1))) {
//         printf("Bad arguments were passed to the exponentialIntegral function
//         call\n"); return static_cast<T>(0);  // or optionally return NaN or
//         -1
//     }

//     if (n == 0) {
//         ans = exp(-x) / x;
//     } else {
//         if (x > static_cast<T>(1.0)) {
//             b = x + static_cast<T>(n);
//             c = big;
//             d = static_cast<T>(1.0) / b;
//             h = d;
//             for (i = 1; i <= maxIterations; ++i) {
//                 a = -static_cast<T>(i) * static_cast<T>(nm1 + i);
//                 b += static_cast<T>(2.0);
//                 d = static_cast<T>(1.0) / (a * d + b);
//                 c = b + a / c;
//                 del = c * d;
//                 h *= del;
//                 if (fabs(del - static_cast<T>(1.0)) <= epsilon) {
//                     ans = h * exp(-x);
//                     return ans;
//                 }
//             }
//             ans = h * exp(-x);
//             return ans;
//         } else {
//             ans = (nm1 != 0) ? static_cast<T>(1.0) / static_cast<T>(nm1) :
//             -log(x) - eulerConstant; fact = static_cast<T>(1.0); for (i = 1;
//             i <= maxIterations; ++i) {
//                 fact *= -x / static_cast<T>(i);
//                 if (i != nm1) {
//                     del = -fact / static_cast<T>(i - nm1);
//                 } else {
//                     psi = -eulerConstant;
//                     for (ii = 1; ii <= nm1; ++ii) {
//                         psi += static_cast<T>(1.0) / static_cast<T>(ii);
//                     }
//                     del = fact * (-log(x) + psi);
//                 }
//                 ans += del;
//                 if (fabs(del) < fabs(ans) * epsilon) return ans;
//             }
//             return ans;
//         }
//     }
//     return ans;
// }

template <typename T>
__global__ void exponential_integral_kernel(
    const double a, const double division, const int maxIterations,
    const int blocks_per_col, const int start_row, const int n,
    const int numberOfSamples, T* results_gpu) {
  int       row_idx     = blockIdx.x + start_row;
  const int block_steps = blocks_per_col;
  int       ui          = row_idx + 1;

  if (row_idx < n) {
    for (int i = 0; i < numberOfSamples; i += block_steps) {
      int uj = i + threadIdx.x + 1;
      if (uj <= numberOfSamples) {
        double x                 = a + uj * division;
        int    glb_idx           = row_idx * numberOfSamples + uj;
        results_gpu[glb_idx - 1] = exponentialIntegral(ui, x, maxIterations);
        // results_gpu[glb_idx-1] = exponentialIntegralDeekSeek(ui, x,
        // maxIterations); results_gpu[glb_idx-1] =
        // exponentialIntegralChatGPT(ui, x, maxIterations);
      }
    }
  }
}

void exponentialIntegralGpu(const unsigned int numberOfSamples,
                            const unsigned int n, const double a,
                            const double division, const int maxIterations,
                            float* resultsGpu, const int block_size,
                            const int stream_num,
                            float     timings[CUDA_STREAMS_MAX]) {
  cudaStream_t streams[stream_num];
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
  if (stream_num > 1) {
    for (int i = 0; i < stream_num; i++) {
      CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    for (int start_row = 0; start_row < n;
         start_row += stream_num * row_per_batch) {
      for (int i = 0; i < stream_num; i++) {
        exponential_integral_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            a, division, maxIterations, blocks_per_col,
            start_row + i * row_per_batch, n, numberOfSamples, results_gpu);
      }
    }
    for (int i = 0; i < stream_num; i++) {
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  } else {
    for (int start_row = 0; start_row < n; start_row += row_per_batch) {
      exponential_integral_kernel<<<grid_size, block_size>>>(
          a, division, maxIterations, blocks_per_col, start_row, n,
          numberOfSamples, results_gpu);
    }
  }
  TIME_END();

  TIME_START();
  CHECK_CUDA(cudaMemcpy(resultsGpu, results_gpu,
                        n * numberOfSamples * sizeof(float),
                        cudaMemcpyDeviceToHost));
  TIME_END();

  CHECK_CUDA(cudaFree(results_gpu));
  if (stream_num > 1)
    for (int i = 0; i < stream_num; i++) {
      CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
  TIME_FINISH();

  return;
}

void exponentialIntegralGpu(const unsigned int numberOfSamples,
                            const unsigned int n, const double a,
                            const double division, const int maxIterations,
                            double* resultsGpu, const int block_size,
                            const int stream_num,
                            float     timings[CUDA_STREAMS_MAX]) {
  cudaStream_t streams[stream_num];
  TIME_INIT();

  TIME_START();
  const int blocks_per_col = div_up(n, block_size);
  const int grid_size      = row_per_batch;
  TIME_END();

  TIME_START();
  double* results_gpu = NULL;
  CHECK_CUDA(cudaMalloc((void**) (&results_gpu),
                        n * numberOfSamples * sizeof(double)));
  TIME_END();

  TIME_START();
  if (stream_num > 1) {
    for (int i = 0; i < stream_num; i++) {
      CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    for (int start_row = 0; start_row < n;
         start_row += stream_num * row_per_batch) {
      for (int i = 0; i < stream_num; i++)
        exponential_integral_kernel<<<grid_size, block_size, 0, streams[i]>>>(
            a, division, maxIterations, blocks_per_col,
            start_row + i * row_per_batch, n, numberOfSamples, results_gpu);
    }
    for (int i = 0; i < stream_num; i++) {
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
  } else {
    for (int start_row = 0; start_row < n; start_row += row_per_batch) {
      exponential_integral_kernel<<<grid_size, block_size>>>(
          a, division, maxIterations, blocks_per_col, start_row, n,
          numberOfSamples, results_gpu);
    }
  }

  TIME_END();

  TIME_START();
  CHECK_CUDA(cudaMemcpy(resultsGpu, results_gpu,
                        n * numberOfSamples * sizeof(double),
                        cudaMemcpyDeviceToHost));
  TIME_END();

  CHECK_CUDA(cudaFree(results_gpu));
  if (stream_num > 1)
    for (int i = 0; i < stream_num; i++) {
      CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
  TIME_FINISH();

  return;
}
