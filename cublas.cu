#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) { if (err != cudaSuccess) { \
    printf("CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(EXIT_FAILURE); }}

#define CUBLAS_CHECK(err) { if (err != CUBLAS_STATUS_SUCCESS) { \
    printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }}

int main() {
    int N = 1024;
    int M = 1024;
    int K = 1024;

    float *h_A = (float *)malloc(N * K * sizeof(float));
    float *h_B = (float *)malloc(K * M * sizeof(float));
    float *h_C = (float *)malloc(N * M * sizeof(float));

    for (int i = 0; i < N * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * M; ++i) h_B[i] = 2.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * M * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    // Create and start CUDA timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Matrix multiplication: C = A * B
    // A (N×K), B (K×M), C (N×M)
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             d_B, M,
                             d_A, K,
                             &beta,
                             d_C, M));

    // Stop CUDA timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("cuBLAS sgemm time: %.3f ms\n", elapsed_ms);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}