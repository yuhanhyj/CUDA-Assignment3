#include "exponentialIntegralGPU.h"

void exponentialIntegralGpu(const unsigned int numberOfSamples,
                            const unsigned int n, const double a,
                            const double                     division,
                            std::vector<std::vector<float>>& resultsGpu,
                            const int block_size, float timings[5]) {}

void exponentialIntegralGpu(const unsigned int numberOfSamples,
                            const unsigned int n, const double a,
                            const double                      division,
                            std::vector<std::vector<double>>& resultsGpu,
                            const int block_size, float timings[5]) {}
