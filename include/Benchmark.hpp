#pragma once

#include <string>
#include <vector>

template <typename T>
class Benchmark {
    protected:
        std::string name;
        int iterations;
        std::vector<double> runtimes;
        
    public:
        Benchmark(const std::string& n, int iters = 10);

        //getters
        std::string getName() const;
        int getIterations() const;
        double getLastRuntime() const;

        //setters
        void setIterations(int iters);

        //derived classes will override this
        virtual void run() = 0;
};

template<typename T>
Benchmark<T>::Benchmark(const std::string& n, int iters) : name(n), iterations(iters), runtimes(iters) {};

template<typename T>
std::string Benchmark<T>::getName() const {return name;}

template<typename T>
int Benchmark<T>::getIterations() const {return iterations;}

template<typename T>
double Benchmark<T>::getLastRuntime() const 
{
    if(runtimes.empty() == false) return runtimes.at(runtimes.size() - 1);
    else return -1;
};
