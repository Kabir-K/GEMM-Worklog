# GEMM-Worklog

This repository contains my worklog on optimizing **SGEMM (Single-precision General Matrix Multiply)** kernels in CUDA.  
The experiments progressively apply different optimization techniques to improve performance over the baseline implementation.

---

## Benchmark Setup
- Matrix size: `M = N = K = 4096`
- Runs per kernel: `20`
- Hardware: NVIDIA GeForce RTX 4060
- Metric: Execution time (ms) and throughput (GFLOPS / TFLOPS)

---

## Kernel Versions & Results

| Kernel | Description                               | Avg Time (ms) | GFLOPS   | TFLOPs |
|--------|-------------------------------------------|---------------|----------|--------|
| **s00** | Baseline (cuBLAS reference)              | 15.127        | 9085.44  | 9.09   |
| **s01** | Naice implementation                     | 191.207       | 718.79   | 0.71   |
| **s02** | Coalesced memory access                  | 152.901       | 898.87   | 0.89   |
| **s03** | Shared Memory Cache-Blocking             | 123.151       | 1116.02  | 1.12   |
| **s04** | 1D Block tiling                          | 41.169        | 3338.39  | 3.34   |
| **s05** | 2D Block tiling                          | 25.056        | 5485.26  | 5.49   |
| **s06** | Vectorized loads                         | 16.325        | 8418.74  | 8.42   |
| **s07** | Decrease Smem Conflict                   | 19.919        | 6899.95  | 6.90   |
| **s10** | Warp tiling                              | 14.177        | 10394.65 | 10.39  |

---

## Optimization Techniques Applied
- ✅ **Coalesced memory access** – aligning memory accesses for higher bandwidth efficiency.
- ✅ **Shared Memory Cache-Blocking** – loading tiles of matrices into **shared memory** so that threads within a block can reuse data.
- ✅ **Vectorized loads** – using wider load/store instructions to reduce memory transactions.  
- ✅ **Block tiling** – dividing computation into thread blocks to improve reuse.  
- ✅ **Increasing arithmetic intensity** – performing more computations per memory access.  
- ✅ **Warp tiling** – mapping work at the warp level to better exploit SIMT execution.  

---

## Notes
- `sgemm00.cu` is a **cuBLAS reference** implementation. Link cublas to compile ( -lcublas ) 
- Subsequent files (`sgemm03.cu`, `sgemm04.cu`, …, `sgemm10.cu`) implement incremental optimization strategies.  
- Results may vary depending on GPU architecture and CUDA version.  

---

## How to Run
Compile with `nvcc`:
```bash
nvcc sgemm06.cu -o s06
./s06
