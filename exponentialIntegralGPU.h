#ifndef _EXPONENTIALINTEGRALGPU_H_
#define _EXPONENTIALINTEGRALGPU_H_

#include <vector>

extern void exponentialIntegralGpu(const unsigned int numberOfSamples,
                                   const unsigned int n, const double a,
                                   const double division,
                                   const int maxIterations, float* resultsGpu,
                                   const int block_size, float timings[5]);

extern void exponentialIntegralGpu(const unsigned int numberOfSamples,
                                   const unsigned int n, const double a,
                                   const double division,
                                   const int maxIterations, double* resultsGpu,
                                   const int block_size, float timings[5]);

#endif // _EXPONENTIALINTEGRALGPU_H_
