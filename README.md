# CUDA-Accelerated Exponential Integral Calculation

## Overview

This project provides a high-performance implementation of the exponential integral function, *E<sub>n</sub>(x)*, accelerated using CUDA. It is designed to compute the function for a wide range of orders (`n`) and sample points (`x`). The primary goal is to compare the performance of a massively parallel GPU implementation against a traditional sequential CPU version.

The program includes mechanisms for performance timing, correctness verification, and configurable test parameters.

## Core Features

* **Dual Implementations**: Contains both a C++ implementation for the CPU and a CUDA implementation for the GPU, allowing for direct performance comparison.
* **Data Precision**: Supports both single-precision (`float`) and double-precision (`double`) computations on both the CPU and GPU.
* **Correctness Checking**: Automatically compares the results from the GPU against the CPU to verify correctness, printing any discrepancies found.
* **Detailed Performance Timing**: Provides a breakdown of execution time, including:
    * Total CPU execution time.
    * Total GPU execution time (for float and double).
    * Detailed GPU operation timings:
        * Grid/block calculation
        * Device memory allocation (`cudaMalloc`)
        * Kernel execution
        * Memory copy from device to host (`cudaMemcpy`)

## Advanced CUDA Features Implemented

This implementation incorporates several advanced CUDA features to enhance performance and code maintainability:

* **CUDA Streams**: To maximize parallelism, the GPU workload is divided and executed across multiple CUDA streams. This allows for the potential overlap of kernel executions, especially on GPUs with the necessary hardware support. The number of streams is configurable at runtime.
* **Global Memory Usage**: The implementation uses GPU global memory to store the final results matrix. Intermediate calculations within the kernel threads rely on registers and local memory.

## How to Compile and Run

### Prerequisites

* NVIDIA CUDA Toolkit (for `nvcc`)
* A CUDA-capable GPU
* `make` and a C++ compiler (`g++`)

### Compilation

To compile the project, simply run the `make` command from the root directory:

```sh
make
```

This will generate an executable file named `exponentialIntegral.out`.

### Execution

Run the program from the command line, specifying options as needed.

```sh
./exponentialIntegral.out [options]
```

**Command-Line Options**

| Flag | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| `-n` | `size` | Sets the maximum order (`n`) for the exponential integral calculations. | 10 |
| `-m` | `size` | Sets the number of samples to be taken in the interval (`a`, `b`). | 10 |
| `-s` | `num` | Sets the number of CUDA streams to use for the GPU computation. | 1 |
| `-a` | `value` | Sets the lower bound of the interval (`a`). | 0.0 |
| `-b` | `value` | Sets the upper bound of the interval (`b`). | 10.0 |
| `-i` | `size` | Sets the number of iterations for the calculation. | 2000000000 |
| `-t` | - | Enables detailed timing output for both CPU and GPU. | Disabled |
| `-v` | - | Activates verbose mode, printing the calculated results. | Disabled |
| `-c` | - | Skips the CPU computation. | Enabled |
| `-g` | - | Skips the GPU computation. | Enabled |
| `-h` | - | Displays the usage message. | - |

## Performance Results

The following sections present the performance results for various problem sizes and stream configurations. The speedup is calculated as (Total CPU Time / GPU Time) for each precision level.

### Scalability Analysis

This table shows how performance scales as the problem size (`n` x `m`) increases for different kernel block sizes. All runs were performed with a single CUDA stream (`-s 1`).

| Block Size | `n` | `m` | CPU Time (s) | GPU Float Time (s) | GPU Double Time (s) | Speedup (vs. Float) | Speedup (vs. Double) |
|:----------:|:---:|:---:|:------------:|:------------------:|:-------------------:|:-------------------:|:--------------------:|
| **256** | 5000 | 5000 | 4.64 | 0.31 | 0.25 | 14.97x | 18.56x |
| **256** | 8192 | 8192 | 12.14 | 0.38 | 0.48 | 31.95x | 25.29x |
| **256** | 16384 | 16384 | 46.81 | 0.78 | 2.70 | 60.01x | 17.34x |
| **256** | 20000 | 20000 | 72.73 | 1.02 | 3.51 | 71.30x | 20.72x |
| **512** | 5000 | 5000 | 4.64 | 0.32 | 0.27 | 14.50x | 17.18x |
| **512** | 8192 | 8192 | 12.17 | 0.38 | 0.50 | 31.92x | 24.34x |
| **512** | 16384 | 16384 | 46.86 | 0.79 | 2.68 | 59.32x | 17.48x |
| **512** | 20000 | 20000 | 68.96 | 1.02 | 3.48 | 67.61x | 19.82x |

**Analysis**: As the problem size increases, the speedup gained from GPU acceleration becomes significantly more pronounced, especially for single-precision (`float`) calculations. This is expected, as larger workloads are able to better saturate the parallel processing capabilities of the GPU. Comparing the two block sizes, there is no significant performance difference for this particular problem and hardware, suggesting that both 256 and 512 are effective choices.

### Stream Performance Analysis

This table shows the impact of using multiple CUDA streams on the largest problem size (`n=20000`, `m=20000`). These tests were all conducted with a **block size of 256**.

| Streams (`-s`) | CPU Time (s) | GPU Float Time (s) | GPU Double Time (s) | Speedup (vs. Float) | Speedup (vs. Double) |
|:--------------:|:------------:|:------------------:|:-------------------:|:-------------------:|:--------------------:|
| 1 | 68.94 | 1.02 | 3.50 | 67.59x | 19.70x |
| 2 | 71.33 | 1.02 | 3.27 | 69.93x | 21.81x |
| 3 | 68.93 | 1.02 | 3.25 | 67.58x | 21.21x |
| 4 | 69.16 | 1.00 | 3.25 | 69.16x | 21.28x |
| 5 | 68.92 | 1.00 | 3.27 | 68.92x | 21.08x |

**Analysis**: Using multiple streams provides a modest performance improvement for double-precision calculations, with the best speedup observed at 2 streams. For single-precision, the kernel execution time is already very fast, so the benefits of overlapping execution with multiple streams are minimal. This suggests that for this specific problem, the computation is not complex enough to fully leverage the advantages of stream-based parallelism beyond 2-4 streams.

## Comparison with an LLM-Generated Solution

As part of the assignment, I was required to compare my implementation against a solution generated by an LLM. I used the free version of ChatGPT for this and placed its generated code in the `llm/` directory for reference.

The comparison highlighted several key differences and showed that the LLM-generated solution, while functional at a basic level, did not meet several of the core and advanced requirements of the assignment.

The main differences are as follows:

* **Project Structure**: I designed my project to be modular, separating the CPU logic, GPU kernels, and utility functions into different files for better organization and scalability. In contrast, the LLM's solution was a single, monolithic `.cu` file that contained all the code.
* **Data Precision**: A key requirement was support for both single (`float`) and double (`double`) precision. My implementation correctly handles both data types, whereas the LLM's code was limited to `float` only.
* **Advanced Features**: I implemented CUDA streams to leverage potential performance gains from overlapping GPU operations. The LLM's solution did not include any use of CUDA streams and relied on default, synchronous execution.
* **Performance Analysis**: To properly evaluate performance, my implementation includes detailed timing for individual GPU operations (memory allocation, kernel execution, and data transfers). The LLM's version only offered a high-level timer for the total execution.
