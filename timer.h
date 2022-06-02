#include <chrono>

std::chrono::steady_clock::time_point g_start;
std::chrono::steady_clock::time_point g_end;

void start_timer() {
    g_start = std::chrono::high_resolution_clock::now();
}

void end_timer() {
    g_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(g_end - g_start);
    printf("Time took %lld milliseconds.\n", duration.count());
}