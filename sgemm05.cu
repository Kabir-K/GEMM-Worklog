#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define N 4096
#define K 4096
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define alpha 1
#define beta 0
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8


__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1) sgemm_5(float *A, float *B, float *C){
    
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    int threadCol = threadIdx.x % (BN/TN);
    int threadRow = threadIdx.x / (BN/TN);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * K + cCol * BN;

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int strideA = blockDim.x / BK;

    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;
    int strideB = blockDim.x / BN;

    float threadResult[TM * TN] = {0};
    float regA[TM] = {0};
    float regB[TN] = {0};

    for(int bid = 0; bid < K; bid += BK){

        for(int offset = 0; offset<BM; offset += strideA){
            As[innerRowA * BK + offset * BK + innerColA] = A[innerRowA * K + offset * K + innerColA];
        }
        for(int offset = 0; offset<BK; offset += strideB){
            Bs[innerRowB * BN + offset * BN + innerColB] = B[innerRowB * N + offset * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;
    
        for(int dotidx=0; dotidx < BK; dotidx++){
            for(int i=0; i<TM; i++){
                regA[i] = As[threadRow * TM * BK + i * BK + dotidx];
            }
            for(int i=0; i<TN; i++){
                regB[i] = Bs[threadCol * TN + i + dotidx * BN];
            }
            for(int i=0 ; i<TM; i++){
                for(int j=0; j<TN; j++){
                    threadResult[i * TN + j] +=
                        regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i<TM; i++){
        for(int j=0; j<TN; j++){
            C[threadRow * TM * N + i * N + threadCol * TN + j] = threadResult[i * TN + j];
        }
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

    dim3 threadsperblock((BM * BN) / (TM * TN));
    dim3 griddim(CEIL_DIV(N,BN),CEIL_DIV(M,BM));

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    for(int i=0; i<3; i++){
        sgemm_5<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double total = 0;
    int runs = 20;
    for(int i=0; i<runs; i++){
        cudaEventRecord(start);
        sgemm_5<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
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
