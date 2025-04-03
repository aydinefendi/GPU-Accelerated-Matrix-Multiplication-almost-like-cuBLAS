#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

__global__ void matrixMulKernel(float *A, float *B, float *C, int N);


int main(int argc, char *argv[])
{
    int N = 10;
    float *A = (float *)malloc(sizeof(float) * N);
    float *B = (float *)malloc(sizeof(float) * N);
    float *C = (float *)malloc(sizeof(float) * N);

    for (int i = 0; i < N; i++)
    {
        A[i] = (rand() % 10) + 1;
        B[i] = (rand() % 10) + 1;
        C[i] = 0;
    }


    float *d_A, *d_B, *d_C;
    cudaError_t err = cudaMalloc((void**)&d_A, sizeof(float) *N);
    CUDA_CHECK(err);
    err = cudaMalloc((void**)&d_B, sizeof(float) *N);
    CUDA_CHECK(err);
    err = cudaMalloc((void**)&d_C, sizeof(float) *N);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_A, A, sizeof(float) *N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_B, B, sizeof(float) *N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    dim3 block(4, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, 1, 1);
    
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    err = cudaMemcpy(C, d_C, sizeof(float) *N, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    printf("\nVector Multiplication Results:\n");
    for (int i = 0; i < N; i++) {
        printf("%f * %f = %f\n", A[i], B[i], C[i]);
    }
    

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);
    return 0;
}

__global__ void matrixMulKernel(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] * B[i];
    }
}
