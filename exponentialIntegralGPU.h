#ifndef _EXPONENTIALINTEGRALGPU_H_
#define _EXPONENTIALINTEGRALGPU_H_

#include <vector>

#define CUDA_STREAMS_MAX 5

extern void exponentialIntegralGpu(const unsigned int numberOfSamples,
                                   const unsigned int n, const double a,
                                   const double division,
                                   const int maxIterations, float* resultsGpu,
                                   const int block_size, const int stream_num,
                                   float timings[CUDA_STREAMS_MAX]);

extern void exponentialIntegralGpu(const unsigned int numberOfSamples,
                                   const unsigned int n, const double a,
                                   const double division,
                                   const int maxIterations, double* resultsGpu,
                                   const int block_size, const int stream_num,
                                   float timings[CUDA_STREAMS_MAX]);

#endif // _EXPONENTIALINTEGRALGPU_H_
