#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer  {
    public:
        string name;
        std::chrono::steady_clock::time_point g_start;
        std::chrono::steady_clock::time_point g_end;

        Timer(string name) {
            this->name = name;
            start();
        }

        void start() {
            g_start = std::chrono::high_resolution_clock::now();
        }

        void stop() {
            g_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(g_end - g_start);
            printf("%s took %lld milliseconds.\n", name.c_str(), duration.count());
        }

};

#endif