#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <hip/hip_runtime.h>
#include <assert.h>
#include <limits>

inline constexpr size_t GRID_SIZE = 1;
inline constexpr size_t BLOCK_SIZE = 1;
inline constexpr size_t UNROLL_FACTOR = 32;
inline constexpr size_t MAX_ITER = 30;
inline constexpr size_t MAX_SIZE = (1UL << MAX_ITER);

struct Node
{
    Node* next;
};

__device__ __forceinline__ void test_get_realtime(int64_t* out) {
    asm volatile (
        "s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=s"(*out)
    );
}

template<typename T>
__device__ Node* buildPChase(Node* __restrict__ d_q, T* __restrict__ d_p, const size_t arraySize)
{
    Node* k = &d_q[d_p[arraySize - 1]];

    for (size_t i = 0; i < arraySize; i++)
        k = k->next = &d_q[d_p[i]];

    return k;
}

//noinline for debugging
__device__ void __noinline__ recordTimes(float* __restrict__ timerArray, size_t index, float& avgLatency)
{
    __builtin_nontemporal_store(avgLatency, &timerArray[index]);
}

template<typename T>
__global__ void pChaseKernelCoarse(Node* __restrict__ d_q, T* __restrict__ d_p, 
T* __restrict__ d_ptrCopy, float* __restrict__ timerArray, 
size_t* invalidCount, const size_t arraySize)
{
    //always launched with a blockSize of 1
    unsigned int tid = threadIdx.x;

    Node* k = buildPChase<T>(d_q, d_p, arraySize);
    Node* ptr = k;
    float avgLatencyNs;
    size_t curr_idx, stride;
    
    for(size_t iter = 0; iter < (MAX_SIZE / arraySize); iter++)
    {
        int64_t stop = 0;
        int64_t start = 0;

        for(size_t i = 0; i < arraySize; i += UNROLL_FACTOR)
        {    

            test_get_realtime(&start);
            #pragma unroll
            for(size_t j = 0; j < UNROLL_FACTOR; j++)
            {
                asm volatile (
                        "global_load_b64 %0, %1, off\n\ts_waitcnt vmcnt(0)"
                    : "=v"(k)    
                    : "v"(&k->next)   // k = k->next is a 64 bit load (64 bit address space)
                );
            }
            test_get_realtime(&stop);
        
            if(k == nullptr)
            {
                d_ptrCopy[blockIdx.x] = tid;
            }

            int64_t latency = stop - start;
            curr_idx = (i/UNROLL_FACTOR);
            stride = (arraySize/UNROLL_FACTOR);

            avgLatencyNs = ( (float)latency / UNROLL_FACTOR ) * 10; //100 Mhz clock
            recordTimes(timerArray, (iter*stride + curr_idx), avgLatencyNs);
            
        }

        //restart pchase from beginning
        k = ptr;
    }
    
}

template<typename T>
__global__ void pChaseKernelFine(Node* __restrict__ d_q, T* __restrict__ d_p, 
T* __restrict__ d_ptrCopy, float* __restrict__ timerArray, 
size_t* invalidCount, const size_t arraySize)
{
    //always launched with a blockSize of 1
    unsigned int tid = threadIdx.x;

    Node* k = buildPChase<T>(d_q, d_p, arraySize);
    Node* ptr = k;
    size_t currError = 0;

    for(size_t iter = 0; iter < (MAX_SIZE / arraySize); iter++)
    {
        int32_t stop = 0;
        int32_t start = 0;

        for(size_t i = 0; i < arraySize; i++)
        {
            asm volatile(
                "s_getreg_b32 %0, hwreg(HW_REG_SHADER_CYCLES, 0, 20)\n\tglobal_load_b64 %1, %3, off\n\ts_waitcnt vmcnt(0)\n\ts_getreg_b32 %2, hwreg(HW_REG_SHADER_CYCLES, 0, 20)"
                : "=s"(start), "=v"(k), "=s"(stop)    
                : "v"(&k->next) // k = k->next is a 64 bit load (64 bit address space)
            );

            if(k == nullptr)
            {
                d_ptrCopy[blockIdx.x] = tid;
            }

            float latency = stop - start;

            if(latency <= 0)
            {
                ++currError;
            }
            else
            {
                recordTimes(timerArray, ((iter*arraySize - (*invalidCount)) + (i - (currError))), latency);
            }
        }
        //restart pchase from beginning
        k = ptr;
        *invalidCount = currError + *invalidCount;
        currError = 0;
    }
    
}

enum class test_type
{
    FINE_GRAINED,
    COARSE_GRAINED
};

template <typename T>
class PChase : public Benchmark<T> {

    public:
        size_t problem_size;
        test_type type = test_type::COARSE_GRAINED;
        PChase(int iters, size_t n) : Benchmark<T>("PChase", iters), problem_size(n) {};

        void run() override
        {   
            assert(problem_size <= MAX_SIZE);

            size_t kB = ( problem_size * sizeof(T) ) / (1 << 10);
            std::cout << "KB: " << kB << " Num elements: " <<  problem_size << std::endl;

            unsigned int *d_copy, *d_p;
            Node* d_q;
            float* d_LatencyTimes;
            size_t* errors;
            HIP_ASSERT(hipMalloc(&errors, sizeof(size_t)));
            

            unsigned int* p = new unsigned int[problem_size];
            std::iota(p, p + problem_size, 0); // Fill p with 0 to N-1
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(p, p + problem_size, g);

            HIP_ASSERT(hipMalloc((void**)&d_p, sizeof(uint32_t) * problem_size));
            HIP_ASSERT(hipMemcpy(d_p, p, problem_size * sizeof(uint32_t), hipMemcpyHostToDevice));

            delete [] p;

            HIP_ASSERT(hipMalloc((void**)&d_q, sizeof(Node) * problem_size)); //contingous memory of Nodes

            size_t totalSamples;
            // size_t singleRunSamples = ( problem_size / UNROLL_FACTOR );

            
            HIP_ASSERT(hipMalloc((void**)&d_copy, GRID_SIZE * sizeof(uint32_t)));

            hipError_t err;

            for(int i = 0; i < this->iterations; i++)
            {
                HIP_ASSERT(hipMemset(errors, 0, sizeof(size_t)));
                if(type == test_type::COARSE_GRAINED)
                {
                    totalSamples = ( (MAX_SIZE / UNROLL_FACTOR ) );
                    HIP_ASSERT(hipMalloc((void**)&d_LatencyTimes, sizeof(float) * totalSamples));
                    HIP_ASSERT(hipMemset(d_LatencyTimes, 0, sizeof(float)*totalSamples));
                    pChaseKernelCoarse<uint32_t><<<GRID_SIZE, BLOCK_SIZE>>>(d_q, d_p, d_copy, d_LatencyTimes, errors, problem_size);
                }
                else
                {
                    totalSamples = MAX_SIZE;
                    HIP_ASSERT(hipMalloc((void**)&d_LatencyTimes, sizeof(float) * totalSamples));
                    HIP_ASSERT(hipMemset(d_LatencyTimes, 0, sizeof(float)*totalSamples));
                    pChaseKernelFine<uint32_t><<<GRID_SIZE, BLOCK_SIZE>>>(d_q, d_p, d_copy, d_LatencyTimes, errors, problem_size);
                }

                err = hipGetLastError();

                if(err != hipSuccess)
                {
                    std::cerr << "Kernel Launch Failed: " << hipGetErrorString(err) <<std::endl;
                }

                HIP_ASSERT(hipDeviceSynchronize());

            }

            size_t h_errors;
            HIP_ASSERT(hipMemcpy(&h_errors, errors, sizeof(size_t), hipMemcpyDeviceToHost));
            std::cout << "Num errors: " << h_errors << std::endl;

            std::vector<float> DeviceReadTimes(totalSamples - h_errors, std::numeric_limits<float>::min());
            HIP_ASSERT(hipMemcpy(DeviceReadTimes.data(), d_LatencyTimes, (totalSamples - h_errors) * sizeof(float), hipMemcpyDeviceToHost));

            std::cout << "Total Samples: " << totalSamples - h_errors << std::endl;

            HIP_ASSERT(hipFree(d_LatencyTimes));

            std::string units = (type == test_type::COARSE_GRAINED) ? "ns" : "clock cycles";

            std::cout << "Avg Load Latency " << "(" << units << ") " << utils::mean(DeviceReadTimes) 
            << " Median ET : " << utils::median(DeviceReadTimes)
            << " SD: " << utils::standardDeviation(DeviceReadTimes) 
            << " Min : " << utils::minValue(DeviceReadTimes)
            << " Max : " << utils::maxValue(DeviceReadTimes) << std::endl;

            HIP_ASSERT(hipFree(d_q));
            HIP_ASSERT(hipFree(d_p));
            HIP_ASSERT(hipFree(d_copy));
            HIP_ASSERT(hipFree(errors));
             

            
        }

};