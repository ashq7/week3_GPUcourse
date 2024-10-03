#include <stdio.h>


const int DSIZE = 50;
const int block_size = 10; //what should this be in the range of? (error earlier was that these were not divisible)
const int grid_size = DSIZE/block_size;


__global__ void vector_swap(float *A, float *B, float *C, int v_size) {

    //FIXME:
    // Express the vector index in terms of threads and blocks
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // Swap the vector elements - make sure you are not out of range
    if (idx < v_size){
        //printf("I'm inside vector_swap!");
        C[idx] = A[idx];
        A[idx] = B[idx];
        B[idx] = C[idx];
    }
}


int main() {

    //allocate memory for host pointers
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
        h_C[i] = 0;
        //question: why are the random numbers normally the same?
    }

    // Print initial elements to check swapping against later
    //question: does it matter if I print from device or host?
    printf("Matrix A: ");
    for (int i = 0; i < DSIZE; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");

    printf("Matrix B: ");
    for (int i = 0; i < DSIZE; i++) {
        printf("%f ", h_B[i]);
    }
    printf("\n");

    // Allocate memory for host and device pointers
    //question: do I need to allocate memory for the host even though I already have above? 
    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));
    cudaMalloc(&d_C, DSIZE*sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

    // Print and check some elements to make sure swapping was successful
    //why does trying to print device values give me a seg fault?
    // printf("Device Matrix A pre-swap: ");
    // for (int i = 0; i < DSIZE; i++) {
    //     printf("%f ", d_A[i]);
    // }
    // printf("\n");

    // printf("Device Matrix B pre-swap: ");
    // for (int i = 0; i < DSIZE; i++) {
    //     printf("%f ", d_B[i]);
    // }
    // printf("\n");

    // Launch the kernel
    vector_swap<<<grid_size, block_size>>>(d_A, d_B, d_C, DSIZE);
    cudaDeviceSynchronize();

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure swapping was successful
    printf("Host Matrix A post-swap: ");
    for (int i = 0; i < DSIZE; i++) {
        printf("%f ", h_A[i]);
    }
    printf("\n");

    printf("Host Matrix B post-swap: ");
    for (int i = 0; i < DSIZE; i++) {
        printf("%f ", h_B[i]);
    }
    printf("\n");

    // Free the memory 
    //question: should host ones be freed?
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //question: do I need:
    //cudaDeviceSynchronize();
    return 0;
}
