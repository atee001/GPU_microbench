#pragma once

#include "Benchmark.hpp"
#include "Utilities.hpp"
#include <iostream>
#include <vector>

template<typename T>
inline void vecAdd(T* out, T* in1, T*in2, const size_t n)
{
    for(size_t i = 0; i < n; i++)
    {
        out[i] = in1[i] + in2[i];
    }
}

template <typename T>
class VectorAdd : public Benchmark<T> {
    
    public:
        size_t problem_size;
        VectorAdd(int iters, size_t n) : Benchmark<T>("VectorAdd", iters), problem_size(n) {} ;

        void run() override
        {   
            std::vector<T> A(problem_size), B(problem_size), C(problem_size);

            utils::Timer t1;

            //warmup
            vecAdd(C.data(), A.data(), B.data(), problem_size);

            for(int i = 0; i < this->iterations; i++)
            {
                t1.reset();
                vecAdd(C.data(), A.data(), B.data(), problem_size);
                auto elapsed = t1.getMilliseconds();
                this->runtimes.at(i) = elapsed;

                std::cout << "Iteration: " << i << " Elapsed (ms): " << elapsed << std::endl;
             }

            std::cout << "Average ET : " << utils::mean(this->runtimes) 
            << " Median ET : " << utils::median(this->runtimes)
            << " SD: " << utils::standardDeviation(this->runtimes) 
            << " Min : " << utils::minValue(this->runtimes)
            << " Max : " << utils::maxValue(this->runtimes) << std::endl;
        }

};