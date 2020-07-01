#include <iostream>
#include <cmath>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

class Node_t { // Turn this into seperate vectors, because cache exists
    public:
        float coordinate_;
        Node_t* neighbour_[2];
        float velocity_;
        float velocity_next_;
};

class Edge_t {
    public:
        Node_t* nodes_[2];
};

__global__
void create_nodes(int n, float *coordinates, float *u_0)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1000;
    float *coordinates;
    float *u_0;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&coordinates, N*sizeof(float));
    cudaMallocManaged(&u_0, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; ++i) {
        coordinates[i] = i * 1.0f/static_cast<float>(N - 1);
        u_0[i] = sin(i * M_PI/static_cast<float>(N - 1));
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    create_nodes<<<numBlocks, blockSize>>>(N, coordinates, u_0);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}