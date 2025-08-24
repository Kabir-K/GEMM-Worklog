#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define N 4096
#define K 4096
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define alpha 1
#define beta 0
#define BM 64
#define BN 64
#define BK 8
#define TM 8


__global__ void sgemm_4(const float *A, const float *B, float *C) {
// If we flip x and y here we get ~30% less performance for large matrices.
// The current, 30% faster configuration ensures that blocks with sequential
// blockIDs access columns of B sequentially, while sharing the same row of A.
// The slower configuration would share columns of A, but access into B would
// be non-sequential. So the faster configuration has better spatial locality
// and hence a greater L2 hit rate.
const uint cRow = blockIdx.y;
const uint cCol = blockIdx.x;

// each warp will calculate 32*TM elements, with 32 being the columnar dim.
const int threadCol = threadIdx.x % BN;
const int threadRow = threadIdx.x / BN;

// allocate space for the current blocktile in SMEM
__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];

// Move blocktile to beginning of A's row and B's column
A += cRow * BM * K;
B += cCol * BN;
C += cRow * BM * N + cCol * BN;

// todo: adjust this to each thread to load multiple entries and
// better exploit the cache sizes

const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
const uint innerRowA = threadIdx.x / BK;
const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
const uint innerRowB = threadIdx.x / BN;

// allocate thread-local cache for results in registerfile
float threadResults[TM] = {0.0};

// outer loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
// populate the SMEM caches
As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
__syncthreads();

// advance blocktile
A += BK;
B += BK * N;

// calculate per-thread results
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
// we make the dotproduct loop the outside loop, which facilitates
// reuse of the Bs entry, which we can cache in a tmp var.
float tmpB = Bs[dotIdx * BN + threadCol];
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
threadResults[resIdx] +=
As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
}
}
__syncthreads();
}

// write out the results
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
C[(threadRow * TM + resIdx) * N + threadCol] =
alpha * threadResults[resIdx] +
beta * C[(threadRow * TM + resIdx) * N + threadCol];
}
}


void initialize(float *a, int size){
    for(int i=0; i<size; i++){
        a[i] = i%1000;
    }
}

int main(){
    float *h_a = (float*) malloc(M * K * sizeof(float));
    float *h_b = (float*) malloc(K * N * sizeof(float));

    initialize(h_a,M*K);
    initialize(h_b,N*K);

    dim3 threadsperblock(BM * BN / TM);
    dim3 griddim(CEIL_DIV(N,BN),CEIL_DIV(M,BM));

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K*N*sizeof(float), cudaMemcpyHostToDevice);


    sgemm_4<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double total = 0;
    int runs = 20;
    for(int i=0; i<runs; i++){
        cudaEventRecord(start);
        sgemm_4<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms,start, stop);    
        total += ms;    
    }
    
    double avg_ms = total / runs;
    double avg_s = avg_ms / 1000;

    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = total_flops / (avg_s * 1.0e9);
    double tflops = gflops/1000;
    
    printf(
        "M=%d N=%d K=%d | runs=%d\n"
        "Avg time = %.3f ms (%.6f s) | GFLOPS = %.2f | TFLOPs = %.2f\n",
        M, N, K, runs,
        avg_ms, avg_s, gflops, tflops
    );


}
