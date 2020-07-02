#include <iostream>
#include <cmath>
// Kernel function to add the elements of two arrays
__global__
void add(int n, float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

class Node_t { // Turn this into seperate vectors, because cache exists
    public:
        __device__ 
        Node_t(float coordinate, Node_t* neighbour0, Node_t* neighbour1, float velocity) 
            : coordinate_(coordinate), neighbour_{neighbour0, neighbour1}, velocity_(velocity), velocity_next_(0.0f) {}

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
void create_nodes(int n, Node_t* nodes, Node_t* boundaries) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (index == 0) {
        boundaries[0] = Node_t(0.0, nullptr, nodes, 0.0f);
        boundaries[1] = Node_t(1.0, nodes + n - 1, nullptr, 0.0f);
    }

    for (int i = index; i < n; i += stride) {
        float coordinate = (i + 1) * 1.0f/static_cast<float>(n + 1);
        float velocity = sin((i + 1) * M_PI/static_cast<float>(n + 1));
        Node_t* neighbour0 = (i > 0) ? (nodes + i - 1) : boundaries;
        Node_t* neighbour1 = (i < (n - 1)) ? (nodes + i + 1) : boundaries + 1;
        nodes[i] = Node_t(coordinate, neighbour0, neighbour1, velocity);
    }
}

__global__
void get_velocity(int n, float* velocity, Node_t* nodes, Node_t* boundaries) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    if (index == 0) {
        velocity[0] = boundaries[0].velocity_;
        velocity[n + 1] = boundaries[1].velocity_;
    }

    for (int i = index; i < n; i += stride) {
        velocity[i + 1] = nodes[i].velocity_;
    }
}

__global__
void timestep(int n, float delta_t, Node_t* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        u = nodes[i].velocity_;
        u_L = nodes[i].neighbour_[0]->velocity_;
        u_R = nodes[i].neighbour_[1]->velocity_;
        r_L = std::abs(nodes[i].coordinate_ - nodes[i].neighbour_[0]->coordinate_);
        r_R = std::abs(nodes[i].coordinate_ - nodes[i].neighbour_[1]->coordinate_);

        nodes[i].velocity_next_ = u * (1 - delta_t * ((u_R - u_L - ((u_R + u_L - 2 * u) * (std::pow(r_R, 2) - std::pow(r_L, 2)))/(std::pow(r_R, 2) + std::pow(r_L, 2)))/(r_R + r_L) 
                    /(1 + (r_R - r_L) * (std::pow(r_R, 2) - std::pow(r_L, 2))/((r_R + r_L) * (std::pow(r_R, 2) + std::pow(r_L, 2))))));
    }
}

__global__
void update(int n, Node_t* nodes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        nodes[i].velocity_ = nodes[i].velocity_next_;
    }
}

int main(void) {
    const int N = 1000;
    float timestep = 0.1;
    Node_t* nodes;
    Node_t* boundaries;

    // Allocate GPU Memory – accessible from GPU
    cudaMalloc(&nodes, N*sizeof(Node_t));
    cudaMalloc(&boundaries, 2*sizeof(Node_t));

    // Run kernel on 1000 elements on the GPU, initializing nodes
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    create_nodes<<<numBlocks, blockSize>>>(N, nodes, boundaries);

    float* velocity;
    cudaMallocManaged(&velocity, (N+2)*sizeof(float));

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    get_velocity<<<numBlocks, blockSize>>>(N, velocity, nodes, boundaries);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    for (int i = 0; i < N+2; ++i) {
        std::cout << "u_" << i << ": " << velocity[i] << std::endl;
    }

    // Free memory
    cudaFree(nodes);
    cudaFree(boundaries);
    cudaFree(velocity);
    
    return 0;
}