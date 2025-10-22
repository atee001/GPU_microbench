#pragma once

#include <chrono>
#include <numeric>
#include <vector>

#define HIP_ASSERT(status) \
    assert(status == hipSuccess)

namespace utils{

    class Timer {
        private:
            std::chrono::high_resolution_clock::time_point startTime;

        public:
            Timer();
            void reset();
            float getMilliseconds() const;
    };

    inline float mean(const std::vector<float>& data);
    inline float standardDeviation(const std::vector<float>& data);
    inline float minValue(const std::vector<float>& data);
    inline float maxValue(const std::vector<float>& data);
    inline float median(const std::vector<float>& data); 

    Timer::Timer() { 
        reset(); 
    }
    void Timer::reset() 
    { 
        startTime = std::chrono::high_resolution_clock::now(); 
    }

    inline float Timer::getMilliseconds() const 
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(endTime - startTime).count();
    }

    inline float mean(const std::vector<float>& data)
    {
        if (data.empty()) return 0.0;
        float sum = std::accumulate(data.begin(), data.end(), 0.0);
        return sum / data.size();
    }

    inline float standardDeviation(const std::vector<float>& data)
    {
        if (data.size() <= 1) return 0.0;

        float avg = mean(data);
        float sq_sum = 0.0;
        for (float x : data) {
            sq_sum += (x - avg) * (x - avg);
        }
        return std::sqrt(sq_sum / (data.size() - 1));

    }

    inline float minValue(const std::vector<float>& data)
    {
        return data.empty() ? 0.0 : *std::min_element(data.begin(), data.end());
    }

    inline float maxValue(const std::vector<float>& data)
    {
        return data.empty() ? 0.0 : *std::max_element(data.begin(), data.end());
    }

    inline float median(const std::vector<float>& data)
    {
        if(data.empty()) return 0.0;
        
        std::vector<float> temp = data;
        std::sort(temp.begin(), temp.end());

        if(temp.size() % 2 == 0)
        {
            return ( temp.at( (temp.size()/2) - 1) + temp.at(temp.size()/2) ) / 2.0;
        }
        else return temp.at(temp.size() / 2);

    }
}