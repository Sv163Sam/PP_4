#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>

__global__ void matrixMult(const double *A, const double *B, double *C, int column_a, int column_b)
{
    int i0 = column_a * (blockDim.y * blockIdx.y + threadIdx.y);
    int j0 = blockDim.x * blockIdx.x + threadIdx.x;
    double sum = 0;

    for (int k = 0; k < column_a; k++)
    {
        sum += A[i0 + k] * B[k * column_b + j0];
    }
    int ind = column_b * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    C[ind] = sum;
}

int mod(int a, int b)
{
    int mod = a % b;
    if (mod != 0)
    {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int row_a = 100;
    int column_a = 200;
    int row_b = 200;
    int column_b = 150;
    
    row_a = mod(row_a, 16);
    std::cout << "A - rows: " << row_a << std::endl;
    column_a = mod(column_a, 16);
    std::cout << "A - columns: " << column_a << std::endl;
    row_b = mod(row_b, 16);
    std::cout << "B - rows: " << row_b << std::endl;
    column_b = mod(column_b, 16);
    std::cout << "B - columns: " << column_b << std::endl;
    
    size_t mtrx_a_size = row_a * column_a * sizeof(double);
    size_t mtrx_b_size = row_b * column_b * sizeof(double);
    size_t mtrx_c_size = row_a * column_b * sizeof(double);
    
    double *h_A = (double*)malloc(mtrx_a_size);
    double *h_B = (double*)malloc(mtrx_b_size);
    double *h_C = (double*)malloc(mtrx_c_size);
    
    for (int i = 0; i < row_a * column_a; ++i)
    {
        h_A[i] = rand()/(double)RAND_MAX;
    }
    for (int i = 0; i < row_b * column_b; ++i)
    {
        h_B[i] = rand()/(double)RAND_MAX;
    }
    
    double *d_A = NULL;
    cudaMalloc((void **)&d_A, mtrx_a_size);
    double *d_B = NULL;
    cudaMalloc((void **)&d_B, mtrx_b_size);
    double * d_C = NULL;
    cudaMalloc((void **)&d_C, mtrx_c_size);
    
    cudaMemcpy(d_A, h_A, mtrx_a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mtrx_b_size, cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock = dim3(16, 16);
    dim3 blocksPerGrid = dim3(column_b / 16, row_a / 16);
    
    cudaEventRecord(start, 0);
    matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, column_a, column_b);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float KernelTime;
    
    cudaEventElapsedTime( &KernelTime, start, stop);
    std::cout << "Time: " << KernelTime << std::endl;
    
    cudaMemcpy(h_C, d_C, mtrx_c_size, cudaMemcpyDeviceToHost);
    //test + print
    for (int i = 0; i < row_a; i++)
    {
        for (int j = 0; j < column_b; j++)
        {
            double sum = 0;
            std::cout << "I - index: " << i << "\tJ - index: " << j << "\tElement: " << h_C[i * column_b + j] << std::endl;
            for (int k = 0; k < column_a; k++)
            {
                sum += h_A[i * column_a + k] * h_B[k * column_b + j];
            }
            if (fabs(h_C[i * column_b + j] - sum) > 1e-3)
            {
                fprintf(stderr, "Result verification failed at element [%d, %d]!\n", i, j);
                exit(EXIT_FAILURE);
            }
        }
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
