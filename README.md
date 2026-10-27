# AKT26 Homework 4: CUDA from zero to hero

Team: l4_p4t1ss3r1e

This is a fork of [abdelfattah83/akt26_gpu_dev](https://github.com/abdelfattah83/akt26_gpu_dev). We wrote the GPU kernels in `3-ger/ger.cu` and `4-gemm/gemm.cu`. The CPU reference, the error checks and the timing code come from the course and are unchanged.

## How to run

Open `colab_run.ipynb` in Google Colab, pick a T4 GPU runtime, and run all cells.

To run on a machine with a GPU instead:

```bash
cd 3-ger  && make sm=75 && ./ger
cd 4-gemm && make sm=75 && ./gemm 16 1000 && ./gemm 32 1000
```

`sm=75` targets the T4. The makefiles default to `sm=80` (A100).

All numbers below come from a Tesla T4 on Colab. Times can change by about 10% from one run to the next, and so can cuBLAS times. Speedup is cuBLAS time divided by our time, so a value above 1 means our kernel is faster.

## GER: `A += alpha * x * y^T`

A is an m x n matrix stored column by column, so `A(i,j) = A[j*m + i]`.

**Kernel.** Each thread updates one element of A. `threadIdx.x` picks the row and `threadIdx.y` picks the column. The 32 threads of a warp get 32 rows next to each other in the same column. Those are 32 floats next to each other in memory, so the GPU reads and writes them in one transaction. An `if (i < m && j < n)` check stops the extra threads in the last blocks from writing outside the matrix.

**Launch.** Blocks are 32 x 8 (256 threads). The grid is `ceil(m/32) x ceil(n/8)`, which is 125 x 250 blocks for the default 4000 x 2000 matrix.

| m x n | Time (µs) | GFLOP/s | Error | cuBLAS time (µs) | cuBLAS GFLOP/s | Speedup |
|---|---|---|---|---|---|---|
| 4000 x 2000 | 272.4 | 58.7 | 4.44e-8 | 303.7 | 52.7 | 1.12x |

Across three runs the speedup ranged from 1.06x to 1.12x. GER reads and writes the whole matrix (64 MB) and does only two math operations per element. That makes memory speed the limit, not math. 64 MB in 272 µs is about 235 GB/s, roughly 73% of the T4's 320 GB/s peak.

## Batched GEMM: `C[b] = A[b] * B[b]`

There are 1000 matrices of size n x n (n is at most 32), stored column by column, one after another.

**Launch (v1 and v2).** We launch one block per matrix, so the grid has 1000 blocks. Each block is n x n threads, which is 256 threads for n=16 and 1024 for n=32. Each thread computes one element of C. As in GER, `threadIdx.x` is the row, so loads of A and stores of C are coalesced. Every thread in a warp reads the same element of B, which the hardware sends to all of them in one read.

**v1, simple.** Each thread runs the k loop directly on global memory.

**v2, shared memory.** Each thread first copies one element of A and one of B into shared memory. After `__syncthreads()`, the k loop reads only from shared memory. Each matrix now leaves global memory once per block instead of once per thread that needs it.

**v3, 4 columns per thread (the version in `gemm.cu` now).** The block shrinks to n x ceil(n/4) threads (64 for n=16, 256 for n=32), and the grid stays at 1000 blocks. Each thread computes 4 elements in one row of C: columns `threadIdx.y`, `threadIdx.y + blockDim.y`, and so on. It reads its element of A from shared memory once per step of the k loop and uses it for all 4 columns. The threads copy A and B into shared memory with a loop, because there are now fewer threads than elements. A `col < n` check covers sizes that are not a multiple of 4.

| Version | n | Time (µs) | GFLOP/s | Max error | cuBLAS time (µs) | cuBLAS GFLOP/s | Speedup |
|---|---|---|---|---|---|---|---|
| v1 simple | 16 | 33.19 | 246.8 | 2.44e-8 | 69.63 | 117.7 | 2.10x |
| v1 simple | 32 | 187.04 | 350.4 | 1.34e-8 | 123.97 | 528.7 | 0.66x |
| v2 shared | 16 | 33.52 | 244.4 | 2.44e-8 | 70.81 | 115.7 | 2.11x |
| v2 shared | 32 | 181.34 | 361.4 | 1.34e-8 | 147.93 | 443.0 | 0.82x |
| v3 4 cols/thread | 16 | 24.45 | 335.0 | 2.44e-8 | 70.74 | 115.8 | 2.89x |
| v3 4 cols/thread | 32 | 94.06 | 696.7 | 1.34e-8 | 129.49 | 506.1 | 1.38x |
| v3 4 cols/thread | 17 | 33.59 | 292.5 | 2.15e-8 | 162.68 | 60.4 | 4.84x |

v1 and v2 are about twice as fast as cuBLAS at n=16 but lose at n=32. v3 is faster than cuBLAS at both sizes. We also ran v3 at n=17 to test a size that is not a multiple of 4. It passes, and cuBLAS is slow at that size.

Shared memory barely changed our time (187 µs to 181 µs at n=32, within run-to-run noise). The T4's L1 cache already kept v1's repeated loads close to the cores, so v2 removed work the cache was already doing. Most of the change in speedup at n=32 comes from cuBLAS running slower in that session (124 µs to 148 µs).

At n=32 a block has 1024 threads, the most a T4 SM can hold, so each SM runs one block at a time. While that block waits on memory, no other block can use the SM. v3 fixes this. With 256 threads per block, 4 blocks fit on each SM, so one block can compute while another waits for its data. Each thread also does 4 multiply-adds per value of A it reads instead of 1. Together these cut the n=32 time from 181 µs to 94 µs.
