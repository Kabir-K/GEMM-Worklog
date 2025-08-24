#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4096
#define N 4096
#define K 4096
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define alpha 1
#define beta 0

template <const int BM = 128, const int BN = 128, const int TM= 8, const int TN = 8, const int BK = 8>
__global__ void sgemm_7(float *A, float *B, float *C){

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BN * BK];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    int threadCol = threadIdx.x % (BN/TN);
    int threadRow = threadIdx.x / (BN/TN);

    int innerRowA = threadIdx.x / (BK/4);
    int innerColA = threadIdx.x % (BK/4);
    int innerRowB = threadIdx.x / (BN/4);
    int innerColB = threadIdx.x % (BN/4);

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for(int bk=0; bk<K; bk+=BK){

        float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColB * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

        tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
        Bs[16 * 0 + innerRowB * 8 * 16 + innerColB/2 + (innerColB % 2) * 4 * 16] = tmp.x;
        Bs[16 * 1 + innerRowB * 8 * 16 + innerColB/2 + (innerColB % 2) * 4 * 16] = tmp.y;
        Bs[16 * 2 + innerRowB * 8 * 16 + innerColB/2 + (innerColB % 2) * 4 * 16] = tmp.z;
        Bs[16 * 3 + innerRowB * 8 * 16 + innerColB/2 + (innerColB % 2) * 4 * 16] = tmp.w;

        __syncthreads();

        A += BK;
        B += BK * N;

        for(int dotIdx=0; dotIdx<BK; dotIdx++){
            for(int i=0; i<TM; i++){
                regM[i] = As[threadRow * TM + dotIdx * BM + i];
            }
            for(int i=0; i<TM; i++){
                regN[i] = Bs[threadCol + i * 16 + dotIdx * 8 * 16];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                  threadResults[resIdxM * TN + resIdxN] +=
                      regM[resIdxM] * regN[resIdxN];
                }
            } 
        }
        __syncthreads();
    }
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(&C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
          tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
          tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
              tmp;
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

    int TM = 8;
    int TN = 8;
    int BM = 128;
    int BN = 128;


    dim3 threadsperblock((BM * BN) / (TM * TN));
    dim3 griddim(CEIL_DIV(N,BN),CEIL_DIV(M,BM));

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));

    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    for(int i=0; i<2; i++){
        sgemm_7<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
        cudaDeviceSynchronize();
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double total = 0;
    int runs = 20;
    for(int i=0; i<runs; i++){
        cudaEventRecord(start);
        sgemm_7<<<griddim,threadsperblock>>>(d_a, d_b, d_c);
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
