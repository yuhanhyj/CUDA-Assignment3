#ifndef _UTILS_GPU_H_
#define _UTILS_GPU_H_

#define CHECK_CUDA(err)                                                        \
  do {                                                                         \
    cudaError_t errors = err;                                                  \
    if (errors != cudaSuccess) {                                               \
      printf("CUDA error: %s\n", cudaGetErrorString(errors));                  \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define TIME_INIT()                                                            \
  cudaEvent_t start;                                                           \
  cudaEvent_t end;                                                             \
  cudaEventCreate(&start);                                                     \
  cudaEventCreate(&end);                                                       \
  int timings_index = 0

#define TIME_START() cudaEventRecord(start)

#define TIME_END()                                                             \
  cudaEventRecord(end);                                                        \
  if (timings != NULL) {                                                       \
    cudaEventSynchronize(start);                                               \
    cudaEventSynchronize(end);                                                 \
    cudaEventElapsedTime(timings + timings_index, start, end);                 \
    timings[timings_index] /= 1000.0f;                                         \
  }                                                                            \
  timings_index++

#define TIME_FINISH()                                                          \
  cudaEventDestroy(start);                                                     \
  cudaEventDestroy(end)

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

inline __host__ __device__ int div_up(int a, int b) {
  if (a < 0) {
    return 0;
  }

  return (a / b) + ((a % b == 0) ? 0 : 1);
}

#endif // _UTILS_GPU_H_
