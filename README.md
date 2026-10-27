# AKT26 Homework 4: CUDA from zero to hero

Team: l4_p4t1ss3r1e

This repo is a fork of [abdelfattah83/akt26_gpu_dev](https://github.com/abdelfattah83/akt26_gpu_dev). We wrote the GPU code in two files:

- `3-ger/ger.cu`
- `4-gemm/gemm.cu`

The course wrote the CPU reference, the error checks and the timing code. We did not change them.

## How to run

Open `colab_run.ipynb` in Google Colab. Pick a T4 GPU runtime, then run all cells.

On your own machine with an NVIDIA GPU:

```bash
cd 3-ger  && make sm=75 && ./ger
cd 4-gemm && make sm=75 && ./gemm 16 1000 && ./gemm 32 1000
```

`sm=75` builds for the T4. Without it, the makefiles build for `sm=80` (the A100).

## How to read the results

- All numbers come from a Tesla T4 on Colab.
- Times change by about 10% from one run to the next. cuBLAS (NVIDIA's own math library) changes just as much.
- GFLOP/s means billions of math operations per second. Higher is better.
- Speedup is cuBLAS time divided by our time. Above 1 means our code is faster.

## GER: `A += alpha * x * y^T`

GER adds the product of two vectors, x and y, to a matrix A. A has m rows and n columns. It is stored one column after another, so element (i, j) sits at `A[j*m + i]`.

### Kernel

Each thread updates one element of A. `threadIdx.x` picks the row and `threadIdx.y` picks the column.

A warp is a group of 32 threads that run together. In our layout, one warp covers 32 rows next to each other in the same column. Those 32 floats sit next to each other in memory, so the GPU reads and writes them in one go.

The last blocks can stick out past the edge of the matrix. The check `if (i < m && j < n)` stops those threads from writing there.

### Launch

- Block: 32 x 8 threads (256 threads).
- Grid: `ceil(m/32) x ceil(n/8)` blocks. For the default 4000 x 2000 matrix, that is 125 x 250 blocks.

### Results

| m x n | Time (µs) | GFLOP/s | Error | cuBLAS time (µs) | cuBLAS GFLOP/s | Speedup |
|---|---|---|---|---|---|---|
| 4000 x 2000 | 272.4 | 58.7 | 4.44e-8 | 303.7 | 52.7 | 1.12x |

Over three runs, the speedup was between 1.06x and 1.12x.

Memory speed limits GER, not math speed. Each call reads and writes the whole matrix (64 MB) but does only two math operations per element. We moved 64 MB in 272 µs, about 235 GB/s. The T4 can move at most 320 GB/s, so we reached about 73% of that.

## Batched GEMM: `C[b] = A[b] * B[b]`

GEMM is matrix multiplication. "Batched" means we multiply 1000 pairs of small matrices in one call. Each matrix is n x n, with n at most 32. Matrices are stored one column after another, and the 1000 matrices sit one after another in memory.

We wrote three versions. `gemm.cu` holds v3. The git history keeps v1 (`029a1fe`) and v2 (`36835ea`).

### v1: simple

- Grid: one block per matrix, so 1000 blocks.
- Block: n x n threads. That is 256 threads for n=16 and 1024 for n=32.
- Each thread computes one element of C. It loops over k and reads A and B straight from global memory (the GPU's main memory).

As in GER, `threadIdx.x` is the row, so a warp reads A and writes C in one go. All threads in a warp need the same element of B, and the GPU sends it to all of them with a single read.

### v2: shared memory

Shared memory is a small, fast memory that all threads in a block can use. In v2, each thread first copies one element of A and one of B into it. After `__syncthreads()`, which waits for every thread to finish copying, the k loop reads only from shared memory.

### v3: 4 columns per thread

- Block: n x ceil(n/4) threads. That is 64 for n=16 and 256 for n=32.
- Grid: still 1000 blocks.
- Each thread computes 4 elements in one row of C: columns `threadIdx.y`, `threadIdx.y + blockDim.y`, and so on.

In each step of the k loop, a thread reads its element of A once and uses it for all 4 columns. There are now fewer threads than elements, so the threads copy A and B into shared memory with a loop. A `col < n` check handles sizes that are not a multiple of 4.

### Results

| Version | n | Time (µs) | GFLOP/s | Max error | cuBLAS time (µs) | cuBLAS GFLOP/s | Speedup |
|---|---|---|---|---|---|---|---|
| v1 simple | 16 | 33.19 | 246.8 | 2.44e-8 | 69.63 | 117.7 | 2.10x |
| v1 simple | 32 | 187.04 | 350.4 | 1.34e-8 | 123.97 | 528.7 | 0.66x |
| v2 shared | 16 | 33.52 | 244.4 | 2.44e-8 | 70.81 | 115.7 | 2.11x |
| v2 shared | 32 | 181.34 | 361.4 | 1.34e-8 | 147.93 | 443.0 | 0.82x |
| v3 4 cols/thread | 16 | 24.45 | 335.0 | 2.44e-8 | 70.74 | 115.8 | 2.89x |
| v3 4 cols/thread | 32 | 94.06 | 696.7 | 1.34e-8 | 129.49 | 506.1 | 1.38x |
| v3 4 cols/thread | 17 | 33.59 | 292.5 | 2.15e-8 | 162.68 | 60.4 | 4.84x |

- v1 and v2 beat cuBLAS by about 2x at n=16, but lose at n=32.
- v3 beats cuBLAS at every size we tried.
- We ran v3 at n=17 to test a size that is not a multiple of 4. It passes. cuBLAS is slow at this size.

### Why v2 barely helped

At n=32, v2 took 181 µs and v1 took 187 µs. That gap is inside the normal run-to-run noise.

The T4 has an L1 cache, a small memory next to each group of cores that keeps recent reads. It already held the values that v1 read again and again. So shared memory did work the cache was already doing.

The speedup still went from 0.66x to 0.82x. Most of that change came from cuBLAS, which ran slower in that session (124 µs, then 148 µs).

### Why v3 helped

The T4 is split into SMs (streaming multiprocessors), and each SM holds at most 1024 threads. At n=32, a v1 or v2 block has 1024 threads. So each SM runs one block at a time. When that block waits for memory, the SM has nothing else to do.

v3 uses 256 threads per block, so 4 blocks fit on each SM. One block can compute while another waits for its data. Each thread also does 4 multiply-adds for each value of A it reads, instead of 1.

Together, these two changes cut the n=32 time from 181 µs to 94 µs.
