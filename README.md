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

The following table will present the performance results for various problem sizes and stream configurations. The speedup is calculated as (Total CPU Time / Total GPU Time).
