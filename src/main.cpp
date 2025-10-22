#include "Benchmark.hpp"
#include "VectorAdd.hpp"
#include "Utilities.hpp"
#include "Pchase.hpp"

#include <iostream>
#include <stdint.h>

constexpr size_t iter = 2; //2 iter one for warmup

int main(int argc, char* argv[])
{
    
    for(uint64_t pSize = 7; pSize < MAX_ITER; pSize++)
    {
        //need to fix this purposely hardcode uint64_t as we are doing pChase on 64 bit addresses, 
        //however the set up uses uint32_t shuffled indices which adds to a base address. 
        //TODO: Need to clean up code
        
        PChase<uint64_t> bench1 = PChase<uint64_t>(iter, 1UL << pSize);
        bench1.run();
    }
 
    return 0;
}