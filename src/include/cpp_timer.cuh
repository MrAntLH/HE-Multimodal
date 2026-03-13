#pragma once

#include <chrono>
#include <atomic>
#include <stdexcept>
#include <cuda_runtime.h> 

class CppTimer {
public:
    explicit CppTimer(std::atomic<long long>* target_total_ms = nullptr,
                      bool enable_cuda_sync = true)
        : m_target_total_ms(target_total_ms),
          m_enable_cuda_sync(enable_cuda_sync) {
        if (m_target_total_ms != nullptr) {
            m_start_time = std::chrono::high_resolution_clock::now();
        }
    }

    ~CppTimer() {
        if (m_target_total_ms != nullptr) {
            auto end_time = std::chrono::high_resolution_clock::now();
            if (m_enable_cuda_sync) {
                cudaError_t cuda_err = cudaDeviceSynchronize();
                if (cuda_err != cudaSuccess) {
                    std::cerr << "[CppTimer 错误] CUDA同步失败！" << std::endl;
                    std::cerr << "  错误码：" << cuda_err << std::endl;
                    std::cerr << "  错误信息：" << cudaGetErrorString(cuda_err) << std::endl;
                    return;
                }
            }

            std::chrono::milliseconds duration_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_time - m_start_time);
            long long single_task_ms = duration_ms.count();

            m_target_total_ms->fetch_add(single_task_ms, std::memory_order_relaxed);
        }
    }

    CppTimer(const CppTimer&) = delete;
    CppTimer& operator=(const CppTimer&) = delete;

private:
    std::chrono::high_resolution_clock::time_point m_start_time;
    std::atomic<long long>* m_target_total_ms;                   
    bool m_enable_cuda_sync;                                     
};