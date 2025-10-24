#pragma once

#include <string>
#include <vector>

template <typename T>
class Bandwidth: public Benchmark<T> 
{
    public:
        size_t problem_size;
        Bandwidth(int iters, size_t n) : Benchmark<T>("Bandwidth", iters), problem_size(n) {};

        void run() override
        {   
            
        }

};