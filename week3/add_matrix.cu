#include <stdio.h>


const int DSIZE_X = 4;
const int DSIZE_Y = 4;

__global__ void add_matrix(float* A, float* B, float* C, int N, int M)
{
    //FIXME:
    // Express in terms of threads and blocks
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idy = blockIdx.x * blockDim.x + threadIdx.x;
    // Add the two matrices - make sure you are not out of range
    //question: why is there no "for" loop?
    if (idx <  N && idy < M ){
        C[idx * N + idy] =  A[idx * N + idy]+B[idx * N + idy];
    }
}

int main()
{

    // Create and allocate memory for host and device pointers 
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE_X * DSIZE_Y];
    h_B = new float[DSIZE_X * DSIZE_Y];
    h_C = new float[DSIZE_X * DSIZE_Y];

    cudaMalloc(&d_A, DSIZE_X*DSIZE_Y*sizeof(float));
    cudaMalloc(&d_B, DSIZE_X*DSIZE_Y*sizeof(float));
    cudaMalloc(&d_C, DSIZE_X*DSIZE_Y*sizeof(float));

    // Fill in the matrices
    // FIXME
    for (int i = 0; i < DSIZE_X; i++) {
        for (int j = 0; j < DSIZE_Y; j++) {
            //FIXME
            h_A[i * DSIZE_X +j] = rand()/(float)RAND_MAX;
            h_B[i * DSIZE_X +j] = rand()/(float)RAND_MAX;
            h_C[i * DSIZE_X +j] = 0;
        }
    }

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE_X*DSIZE_Y*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE_X*DSIZE_Y*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, DSIZE_X*DSIZE_Y*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    // dim3 is a built in CUDA type that allows you to define the block 
    // size and grid size in more than 1 dimentions
    // Syntax : dim3(Nx,Ny,Nz)
    int blockSize_x=1;
    int blockSize_y=1;
    dim3 blockSize(blockSize_x,blockSize_y); 
    dim3 gridSize(DSIZE_X/blockSize_x,DSIZE_Y/blockSize_y); //can I access blockSize entries?
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE_X, DSIZE_Y);

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE_X*DSIZE_Y*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE_X*DSIZE_Y*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, DSIZE_X*DSIZE_Y*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make the addition was succesfull
    printf("Matrix A: ");
    for (int i = 0; i < DSIZE_X*DSIZE_Y; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");

    printf("Matrix B: ");
    for (int i = 0; i < DSIZE_X*DSIZE_Y; i++) {
        printf("%f ", h_B[i]);
    }
    printf("\n");

    printf("Matrix A + Matrix B: ");
    for (int i = 0; i < DSIZE_X*DSIZE_Y; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // Free the memory     
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}