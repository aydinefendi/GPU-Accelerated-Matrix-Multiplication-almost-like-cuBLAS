#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

#define TILE_WIDTH 4

__global__ void matrixMulKernel(float *A, float *B, float *C, int N);

int main(int argc, char *argv[])
{
    int N = 12;
    float *A = (float *)malloc(sizeof(float) * N *N);
    float *B = (float *)malloc(sizeof(float) * N *N);
    float *C = (float *)malloc(sizeof(float) * N *N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = (rand() % 10) + 1;
            B[i*N + j] = (rand() % 10) + 1;
            C[i*N + j] = 0;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaError_t err = cudaMalloc((void**)&d_A, sizeof(float) *N*N);
    CUDA_CHECK(err);
    err = cudaMalloc((void**)&d_B, sizeof(float) *N*N);
    CUDA_CHECK(err);
    err = cudaMalloc((void**)&d_C, sizeof(float) *N*N);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_A, A, sizeof(float) *N*N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_B, B, sizeof(float) *N*N, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    dim3 block(4, 4, 1);
    dim3 grid(ceil(N/block.x), ceil(N/block.y), 1);
    
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N);

    err = cudaMemcpy(C, d_C, sizeof(float) *N*N, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    printf("\nMatrix Multiplication Results:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f * %.2f = %.2f\n", A[i*N + j], B[i*N + j], C[i*N + j]);
        }
        printf("\n");
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
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    assert(N % TILE_WIDTH == 0);

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int i = by * TILE_WIDTH + ty;
    int j = bx * TILE_WIDTH + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;

    for (int phase = 0; phase < (float) ceil(N/TILE_WIDTH);phase++)
    {
        if (i < N && (phase *TILE_WIDTH + tx) < N)
        {
            sh_A[ty][tx] = A[(i*N + (phase *TILE_WIDTH + tx))];
        }
        else 
        {
            sh_A[ty][tx] = 0.0f;
        }

        if (j < N && ((phase*TILE_WIDTH + ty)*N + j) < N)
        {
            sh_B[ty][tx] = B[((phase*TILE_WIDTH + ty)*N + j)];
        }
        else
        {
            sh_B[ty][tx] = 0.0f;
        }
        __synchthreads;

        for (int k = 0; k < TILE_WIDTH;k++)
        {
            value += sh_A[ty][k] * sh_B[k][tx];
        }
        __synchthreads;
    }

    if (i < N && j < N)
    {
        C[i*N+j] = value;
    }
}
