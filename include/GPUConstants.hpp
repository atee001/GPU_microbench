#pragma once

struct Radeon7900XT 
{
    static constexpr size_t num_cus = 84;

    struct Memory {
        static constexpr size_t vCache   = 32 * (1 << 10);
        static constexpr size_t L1       = 256 * (1 << 10);
        static constexpr size_t L2       = 6 * (1 << 20);
        static constexpr size_t INFCACHE = 81920 * (1 << 10);
        static constexpr size_t GDDR6    = 24ULL * (1UL << 30);
    };
};

struct MI300 
{
    static constexpr size_t num_cus = 120;

    struct Memory {
        static constexpr size_t vCache   = 64 * (1 << 10);
        static constexpr size_t L1       = 512 * (1 << 10);
        static constexpr size_t L2       = 12 * (1 << 20);
        static constexpr size_t INFCACHE = 163840 * (1 << 10);
        static constexpr size_t HBM      = 128ULL * (1UL << 30);
    };
};

enum class Benchmarks
{
    PChase,
    Bandwidth,
    VectorAdd,
    MAX_BENCHMARKS
};

template <Benchmarks B, typename GPU>
struct GPUConfig
{
    static constexpr size_t BLOCK_SIZE     = 256; // default
    static constexpr size_t GRID_SIZE      = 1;   // default
    static constexpr size_t UNROLL_FACTOR  = 1;   // default
};


template <typename GPU>
struct GPUConfig<Benchmarks::PChase, GPU>
{
    static constexpr size_t BLOCK_SIZE    = 512;
    static constexpr size_t GRID_SIZE     = 1;
    static constexpr size_t UNROLL_FACTOR = 32;
};

template <typename GPU>
struct GPUConfig<Benchmarks::Bandwidth, GPU>
{
    static constexpr size_t BLOCK_SIZE    = 512;
    static constexpr size_t GRID_SIZE     = 10000 * GPU::num_cus;
    static constexpr size_t UNROLL_FACTOR = 16;
};
