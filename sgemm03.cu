#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define N 4096
#define K 4096
#define BLOCKSIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define alpha 1
#define beta 0

__global__ void sgemm_3(const float *A, const float *B, float *C) {
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;
    __shared__ float As[32 * 32];
    __shared__ float Bs[32 * 32];
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;
    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;
    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
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
    dim3 threadsperblock(BLOCKSIZE * BLOCKSIZE);
    dim3 griddim(CEIL_DIV(M,BLOCKSIZE),CEIL_DIV(N,BLOCKSIZE));

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    
    cudaMemcpy(d_a, h_a, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K*N*sizeof(float), cudaMemcpyHostToDevice);

    for(int i=0; i<3; i++){
        sgemm_3<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double total = 0;
    int runs = 20;
    for(int i=0; i<runs; i++){
        cudaEventRecord(start);
        sgemm_3<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
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
