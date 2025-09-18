#pragma once

#include <chrono>
#include <numeric>
#include <vector>

namespace utils{

    class Timer {
        private:
            std::chrono::high_resolution_clock::time_point startTime;

        public:
            Timer();
            void reset();
            double getMilliseconds() const;
    };

    inline double mean(const std::vector<double>& data);
    inline double standardDeviation(const std::vector<double>& data);
    inline double minValue(const std::vector<double>& data);
    inline double maxValue(const std::vector<double>& data);
    inline double median(const std::vector<double>& data); 

    Timer::Timer() { 
        reset(); 
    }
    void Timer::reset() 
    { 
        startTime = std::chrono::high_resolution_clock::now(); 
    }

    inline double Timer::getMilliseconds() const 
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(endTime - startTime).count();
    }

    inline double mean(const std::vector<double>& data)
    {
        if (data.empty()) return 0.0;
        double sum = std::accumulate(data.begin(), data.end(), 0.0);
        return sum / data.size();
    }

    inline double standardDeviation(const std::vector<double>& data)
    {
        if (data.size() <= 1) return 0.0;

        double avg = mean(data);
        double sq_sum = 0.0;
        for (double x : data) {
            sq_sum += (x - avg) * (x - avg);
        }
        return std::sqrt(sq_sum / (data.size() - 1));

    }

    inline double minValue(const std::vector<double>& data)
    {
        return data.empty() ? 0.0 : *std::min_element(data.begin(), data.end());
    }

    inline double maxValue(const std::vector<double>& data)
    {
        return data.empty() ? 0.0 : *std::max_element(data.begin(), data.end());
    }

    inline double median(const std::vector<double>& data)
    {
        if(data.empty()) return 0.0;
        
        std::vector<double> temp = data;
        std::sort(temp.begin(), temp.end());

        if(temp.size() % 2 == 0)
        {
            return ( temp.at( (temp.size()/2) - 1) + temp.at(temp.size()/2) ) / 2.0;
        }
        else return temp.at(temp.size() / 2);

    }
}