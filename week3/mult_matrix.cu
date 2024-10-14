#include <stdio.h>
#include <time.h>

const int DSIZE = 4;
const float A_val = 3.0f;
const float B_val = 2.0f;

// error checking macro
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)

// Square matrix multiplication on CPU : C = A * B
void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
  //FIXME:
  // i iterates over rows of matrix A
  for (int i = 0; i<size; i++){
    // j iterates over columns of matrix B
    for (int j = 0; j<size; j++){
        float temp = 0;
        // k indexes which item in the ith row of A and jth column of B we are multiplying
        for (int k = 0; k<size; k++){
            //i is analagous to idx, j to idy, size to n
            temp += A[i * size + k] * B [k * size + j];
        }
    C[i*size + j]= temp;
    }
  }
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {

    //FIXME:
    // create thread x index
    // create thread y index
    int idx = blockIdx.y * blockDim.y + threadIdx.y;
    int idy = blockIdx.x * blockDim.x + threadIdx.x;;
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) {
        float temp = 0;
        for (int i = 0; i < size; i++){
            //FIXME : Add dot product of row and column
            temp += A [idx * size +idy] * B [idy * size +idx];
        }
        C[idx*size+idy] = temp;                    
    }

}

int main() {

    float *h_A, *h_B, *h_C_GPU,*h_C_CPU, *d_A, *d_B, *d_C;

    // These are used for timing
    clock_t t0, t1, t2, t3;
    double t1sum=0.0;
    double t2sum=0.0;
    double t3sum=0.0;

    // start timing
    t0 = clock();

    // N*N matrices defined in 1 dimension
    // If you prefer to do this in 2-dimensions, update accordingly
    h_A = new float[DSIZE*DSIZE];
    h_B = new float[DSIZE*DSIZE];
    h_C_GPU = new float[DSIZE*DSIZE];
    h_C_CPU = new float[DSIZE*DSIZE];
    for (int i = 0; i < DSIZE*DSIZE; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C_GPU[i] = 0;
        h_C_CPU[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data from host to device
    cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
    //FIXME:Add all other allocations and copies from host to device
    cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
    cudaCheckErrors("After Memory Allocation");

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C_GPU, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("After copying from host to device");

    // Launch kernel
    // Specify the block and grid dimentions 
    const int block_size = 1;
    dim3 block(1,1);  //FIXME
    dim3 grid(DSIZE/block_size,DSIZE/block_size); //FIXME
    matrix_mul_gpu<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("After launching kernel");

    // Copy results back to host
    cudaMemcpy(h_C_GPU, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("After copying from device back to host");
    // Print and check some elements to make the addition was succesfull
    printf("GPU \n");
    printf("Matrix A: ");
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");

    printf("Matrix B: ");
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        printf("%f ", h_B[i]);
    }
    printf("\n");

    printf("Matrix A * Matrix B: ");
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        printf("%f ", h_C_GPU[i]);
    }
    printf("\n");

    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    // FIXME
    // Excecute and time the cpu matrix multiplication function
    matrix_mul_cpu(h_A,h_B,h_C_CPU,DSIZE);
    // Print and check some elements to make the addition was succesfull
    printf("CPU \n");
    printf("Matrix A: ");
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");

    printf("Matrix B: ");
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        printf("%f ", h_B[i]);
    }
    printf("\n");

    printf("Matrix A * Matrix B: ");
    for (int i = 0; i < DSIZE*DSIZE; i++) {
        printf("%f ", h_C_CPU[i]);
    }
    printf("\n");

    // CPU timing
    t3 = clock();
    t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t3sum);

    // FIXME
    // Free memory 
    free(h_A);
    free(h_B);
    free(h_C_GPU);
    free(h_C_CPU);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
