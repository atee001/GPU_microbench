#include "Benchmark.hpp"
#include "VectorAdd.hpp"
#include "Utilities.hpp"
#include "Pchase.hpp"

#include <iostream>

constexpr size_t iter = 1;

int main(int argc, char* argv[])
{

    for(size_t pSize = 10; pSize < 30; pSize++)
    {
        PChase<size_t>* bench1 = new PChase<size_t>(iter, 1UL << pSize);
        bench1->run();
        delete bench1;
    }
    // VectorAdd<float>* bench1 = new VectorAdd<float>(10, 1 << 24);
    // std::cout << "Benchmark Name: " << bench1->getName() << std::endl;
    // bench1->run();


    // delete bench1;
    return 0;
}