#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <atomic>  

class CudaTimer {
public:
    explicit CudaTimer(std::atomic<long long>* target_duration = nullptr) 
        : m_target_duration(target_duration), m_single_float_duration(0.0f) {
        if (m_target_duration != nullptr) {
            cudaError_t err1 = cudaEventCreate(&m_start);
            cudaError_t err2 = cudaEventCreate(&m_stop);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                throw std::runtime_error("CUDA Event创建失败！错误码：" + std::to_string(err1) + "/" + std::to_string(err2));
            }
            cudaEventRecord(m_start, 0);
        }
    }
    
    ~CudaTimer() {
        if (m_target_duration != nullptr) {
            cudaEventRecord(m_stop, 0);
            cudaEventSynchronize(m_stop);
            cudaEventElapsedTime(&m_single_float_duration, m_start, m_stop);
            long long single_duration = static_cast<long long>(m_single_float_duration);
            m_target_duration->fetch_add(single_duration, std::memory_order_relaxed);
            cudaEventDestroy(m_start);
            cudaEventDestroy(m_stop);
        }
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

private:
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_stop = nullptr;
    std::atomic<long long>* m_target_duration = nullptr;
    float m_single_float_duration;
};