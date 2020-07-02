#include <iostream>
#include <cmath>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

class Node_t { // Turn this into seperate vectors, because cache exists
    public:
        __device__ 
        Node_t(float coordinate, Node_t* neighbour0, Node_t* neighbour1, float velocity) 
            : coordinate_(coordinate), neighbour_{neighbour0, neighbour1}, velocity_(velocity), velocity_next_(0.0) {}

        float coordinate_;
        Node_t* neighbour_[2];
        float velocity_;
        float velocity_next_;
};

class Edge_t {
    public:
        __device__ 
        Edge_t(Node_t* node0, Node_t* node1) : nodes_{node0, node1} {}

        Node_t* nodes_[2];
};

__global__
void create_nodes(int n, Node_t* nodes, Node_t* boundaries)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (index == 0) {
        boundaries[0] = Node_t(0.0, nullptr, nodes, 0.0);
        boundaries[1] = Node_t(1.0, nodes + n - 1, nullptr, 0.0);
    }

    for (int i = index; i < n; i += stride) {
        float coordinate = (i + 1) * 1.0f/static_cast<float>(n + 1);
        float velocity = sin((i + 1) * M_PI/static_cast<float>(n + 1));
        Node_t* neighbour0 = (i > 0) ? (nodes + i - 1) : boundaries;
        Node_t* neighbour1 = (i < (n - 1)) ? (nodes + i + 1) : boundaries + 1;
        nodes[i] = Node_t(coordinate, neighbour0, neighbour1, velocity);
    }
}

int main(void)
{
    const int N = 1000;
    Node_t* nodes;
    Node_t* boundaries;

    // Allocate GPU Memory – accessible from GPU
    cudaMalloc(&nodes, N*sizeof(Node_t));
    cudaMalloc(&boundaries, 2*sizeof(Node_t));


    // Run kernel on 1000 elements on the GPU, initializing nodes
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    create_nodes<<<numBlocks, blockSize>>>(N, nodes, boundaries);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    /*float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;*/

    // Free memory
    cudaFree(nodes);
    cudaFree(boundaries);
    
    return 0;
}