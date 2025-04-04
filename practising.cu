#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <cuda_runtime.h>
#include <math.h>
#include <assert.h>

#define CUDA_CHECK(err) {if (err != cudaSuccess){printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);exit(EXIT_FAILURE);}}

#define TILE_WIDTH 16

__global__ void matrixMulKernel(float *A, float *B, float *C, int N1, int N2, int N3);

int main(int argc, char *argv[])
{
    int N1 = 1024;
    int N2 = 1024; 
    int N3 = 1024;
    float *A = (float *)malloc(sizeof(float) * N1 *N2);
    float *B = (float *)malloc(sizeof(float) * N2 *N3);
    float *C = (float *)malloc(sizeof(float) * N1 *N3);

    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N2; j++) {
            A[i*N2 + j] = (rand() % 10) + 1;
        }
    }

    for (int i = 0; i < N2; i++)
    {
        for (int j = 0; j < N3; j++)
        {
            B[i*N3 + j] = (rand() % 10) + 1;
        }
    }

    for (int i = 0; i < N1; i++)
    {
        for (int j = 0; j < N3; j++)
        {
            C[i*N3 + j] = 0;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaError_t err = cudaMalloc((void**)&d_A, sizeof(float) *N1*N2);
    CUDA_CHECK(err);
    err = cudaMalloc((void**)&d_B, sizeof(float) *N2*N1);
    CUDA_CHECK(err);
    err = cudaMalloc((void**)&d_C, sizeof(float) *N1*N3);
    CUDA_CHECK(err);

    err = cudaMemcpy(d_A, A, sizeof(float) *N1*N2, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);
    err = cudaMemcpy(d_B, B, sizeof(float) *N2*N1, cudaMemcpyHostToDevice);
    CUDA_CHECK(err);

    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 grid(ceil((float) N3/TILE_WIDTH), ceil((float) N1/TILE_WIDTH), 1);

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, N1, N2, N3);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel execution time: %.3f milliseconds\n", elapsedTime);
    
    err = cudaMemcpy(C, d_C, sizeof(float) *N1*N3, cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);
    return 0;
}

__global__ void matrixMulKernel(float *A, float *B, float *C, int N1, int N2, int N3)
{
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int i = by * TILE_WIDTH + ty;
    int j = bx * TILE_WIDTH + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value = 0;

    for (int phase = 0; phase < ceil((float) (N1/TILE_WIDTH));phase++)
    {
        if (i < N1 && (phase *TILE_WIDTH + tx) < N1)
        {
            sh_A[ty][tx] = A[(i*N1 + (phase *TILE_WIDTH + tx))];
        }
        else 
        {
            sh_A[ty][tx] = 0.0f;
        }

        if (j < N2 && ((phase*TILE_WIDTH + ty)*N2 + j) < N2)
        {
            sh_B[ty][tx] = B[((phase*TILE_WIDTH + ty)*N2 + j)];
        }
        else
        {
            sh_B[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH;k++)
        {
            value += sh_A[ty][k] * sh_B[k][tx];
        }
        __syncthreads();
    }

    if (i < N1 && j < N3)
    {
        C[i*N3+j] = value;
    }
}
