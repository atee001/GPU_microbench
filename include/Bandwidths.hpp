#pragma once

#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include "GPUConstants.hpp"

struct dType
{
    uint32_t x[4]; //4 dwords 
};

template<typename T, size_t pSize, size_t NUM_ITER>
__global__ void bandwidth_bench(T* in, T* dummy)
{
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    T reg;

    #pragma unroll
    for(unsigned int iter = 0; iter < NUM_ITER; iter++)
    {
        #pragma unroll
        for(size_t i = 0; i < pSize; i += 512)
        {
            T* addr = &in[(tid + iter + i) % pSize];
            asm volatile("global_load_b128 %0, %1, off" : "=v"(reg) : "v"(addr));
        }
    }
    
    asm volatile("s_waitcnt vmcnt(0)");

    if (reg.x[0] == 1) {
        *dummy = reg; // or some scalar operation
    }
}

template<typename T, size_t NUM_ITER, size_t pSize>
void run_kernel(T* d_in, T* d_dummy, size_t threadsPerBlock, size_t blocksPerGrid)
{
    hipEvent_t start, stop;
    HIP_ASSERT(hipEventCreate(&start));
    HIP_ASSERT(hipEventCreate(&stop));

    HIP_ASSERT(hipEventRecord(start));
    bandwidth_bench<T, pSize / sizeof(T), NUM_ITER><<<blocksPerGrid, threadsPerBlock>>>(d_in, d_dummy);
    HIP_ASSERT(hipEventRecord(stop));

    HIP_ASSERT(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_ASSERT(hipEventElapsedTime(&ms, start, stop));

    size_t bytes_moved = (pSize / threadsPerBlock) * sizeof(T) * threadsPerBlock * blocksPerGrid * NUM_ITER;
    double bandwidth = static_cast<double>(bytes_moved) / (ms / 1000.0) / (1UL << 30);

    std::cout << "Problem size: " << pSize / 1024 << " KB, Bandwidth: "
              << bandwidth << " GB/s" << std::endl;

    HIP_ASSERT(hipEventDestroy(start));
    HIP_ASSERT(hipEventDestroy(stop));
    
}

template <typename T>
class Bandwidth: public Benchmark<T> 
{
    public:
        size_t problem_size;
        Bandwidth(int iters, size_t n) : Benchmark<T>("Bandwidth", iters), problem_size(n) {};

        void run() override
        {   

            T* d_in;
            T* d_dummy;

            constexpr size_t max_bytes = Radeon7900XT::Memory::GDDR6;
            constexpr size_t line_size = 128; 
            constexpr size_t aligned_bytes = (max_bytes / line_size) * line_size;

            constexpr size_t max_elements = (aligned_bytes / sizeof(T))/2; 

            HIP_ASSERT(hipMalloc(&d_in, max_elements * sizeof(T)));
            HIP_ASSERT(hipMalloc(&d_dummy, sizeof(T)));
            
            HIP_ASSERT(hipMemset(d_in, 0xFFFFFFFF, max_elements * sizeof(T)));
            HIP_ASSERT(hipMemset(d_dummy, 0, sizeof(T)));


            constexpr size_t threadsPerBlock = GPUConfig<Benchmarks::Bandwidth, Radeon7900XT>::BLOCK_SIZE;
            constexpr size_t blocksPerGrid = GPUConfig<Benchmarks::Bandwidth, Radeon7900XT>::GRID_SIZE;
            constexpr size_t num_iter = 1e4;

            run_kernel<T, num_iter, 1UL << 10>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 11>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 12>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 13>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 14>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 15>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 16>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 17>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 18>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 19>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 20>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 21>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 22>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 23>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 24>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 25>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 26>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 27>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 28>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 29>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 30>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 31>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 32>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 33>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);
            // run_kernel<T, num_iter, 1UL << 34>(d_in, d_dummy, threadsPerBlock, blocksPerGrid);

            HIP_ASSERT(hipFree(d_in));
            HIP_ASSERT(hipFree(d_dummy));
        }
            

};