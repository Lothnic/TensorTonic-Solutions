#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* input, float* output, int N) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if(i<N){
        output[i] = 0.5*input[i]*(1+tanh(sqrt(2/M_PI)*(input[i]+0.044715*pow(input[i],3))));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
