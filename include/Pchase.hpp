#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

template<typename T>
inline void pointerChase(T* in1, double* timerArray, const size_t n)
{
    T temp;
    int64_t start, stop;
    for(size_t i = 0; i < n; i++)
    {
        T* addr = &in1[i];
        __asm__ volatile("mrs %0, cntvct_el0 \n\tldr %1, [%2]\n\tmrs %3, cntfrq_el0" : "=r"(start), "=r"(temp), "=r"(stop) : "r"(addr) : "memory");
        timerArray[i] = stop - start;
    }
}

template <typename T>
class PChase : public Benchmark<T> {
    
    public:
        size_t problem_size;
        PChase(int iters, size_t n) : Benchmark<T>("PChase", iters), problem_size(n) {};

        void run() override
        {   
            std::vector<T> in(problem_size);
            std::vector<double> timerArray(problem_size);

            T* temp = new T[problem_size];

            std::iota(temp, temp + problem_size, 0);
            std::random_shuffle(temp, temp + problem_size);

            int k = temp[problem_size - 1];
            for (size_t i = 0; i < problem_size; i++)
                k = in[k] = temp[i];

            delete [] temp;

            utils::Timer t1;

            pointerChase(in.data(), timerArray.data(), problem_size);

            for(int i = 0; i < this->iterations; i++)
            {
                t1.reset();
                // vecAdd(C.data(), A.data(), B.data(), problem_size);
                pointerChase(in.data(), timerArray.data(), problem_size);
                auto elapsed = t1.getMilliseconds();
                this->runtimes.at(i) = elapsed;

                std::cout << "Iteration: " << i << "Total Elapsed (ms): " << elapsed << std::endl;

                std::cout << "Average Load Latency: " << utils::mean(timerArray) 
                << " Median ET : " << utils::median(timerArray)
                << " SD: " << utils::standardDeviation(timerArray) 
                << " Min : " << utils::minValue(timerArray)
                << " Max : " << utils::maxValue(timerArray) << std::endl;
             }

            
        }

};