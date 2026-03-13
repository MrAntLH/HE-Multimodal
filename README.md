# HE-Multimodal: 基于 HEonGPU 的全同态多模态模型推理框架

## 项目简介

本项目是一个基于**全同态加密 (FHE)** 的隐私保护多模态模型推理框架，底层依托 [HEonGPU](https://github.com/your-org/HEonGPU) 实现 CUDA 硬件加速，采用 CKKS 同态加密方案，支持浮点数密态运算。框架原生适配 GroundingDINO 等多模态目标检测模型，同时采用通用设计可扩展至其他深度学习网络，整体基于 C++/CUDA 开发。

## 主要特性

- **硬件级加速**: 基于 HEonGPU 的 CUDA 优化，大幅提升同态矩阵乘法、非线性函数近似等核心运算效率
- **CKKS 同态方案**: 支持浮点数加密与运算，天然适配深度学习模型的数值特性
- **模块化封装**: 提供线性层（矩阵乘法）、非线性函数近似、文件 IO、HEonGPU 底层封装等独立模块
- **通用模型架构**: 以 GroundingDINO 为验证场景，结构设计可扩展至其他多模态/单模态模型
- **多维度测试**: 包含端到端模型推理、线性层、非线性层、性能基准等 5 项测试用例

## 目录结构

```
.
├── CMakeLists.txt       # CMake 构建配置（含测试开关控制）
├── data/                # 测试数据目录
│   ├── TextBackbone/    # 文本骨干网络输入/权重（CSV 格式，用于准确率验证）
│   └── ImageBackbone/   # 图像骨干网络输入/权重（CSV 格式，用于准确率验证）
└── src/                 # 源代码目录
    ├── include/         # 头文件 (*.cuh)
    ├── lib/             # 实现文件 (*.cu)
    │   ├── linear/      # 线性函数（矩阵乘法）同态实现
    │   ├── nonlinear/   # 非线性函数（如 ReLU、Sigmoid）多项式近似
    │   ├── file_handler.cu  # CSV 文件读写模块
    │   └── ckks_evaluator.cu # HEonGPU 底层操作封装
    ├── modelTest.cu     # 端到端多模态模型推理测试
    ├── linearTest.cu    # 线性层单独测试
    ├── nonlinearTest.cu # 非线性层近似测试
    ├── benchmarkTest.cu # 性能基准测试
    └── SwinModelTest.cu # Swin Transformer 专用测试
```

## 依赖环境

- **CMake**: >= 3.26
- **C++ Standard**: C++17
- **CUDA Toolkit**: 需与当前 GPU 驱动兼容
- **HEonGPU**: >= 1.1（需提前安装并确保 CMake 可找到 `HEonGPUConfig.cmake`）

## 编译步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-org/he-multimodal.git
   cd he-multimodal
   ```

2. **配置编译选项**
   默认仅启用 `modelTest`，可通过 CMake 选项按需开启其他测试：
   ```bash
   mkdir build && cd build
   cmake .. \
     -DBUILD_MODEL_TEST=ON \          # 默认：启用端到端测试
     -DBUILD_LINEAR_TEST=OFF \        # 按需开启线性层测试
     -DBUILD_NONLINEAR_TEST=OFF \     # 按需开启非线性层测试
     -DBUILD_BENCHMARK_TEST=OFF \     # 按需开启性能基准测试
     -DBUILD_SWINMODEL_TEST=OFF \     # 按需开启 Swin 模型测试
     -DCMAKE_CUDA_ARCHITECTURES=89    # 根据 GPU 调整架构（如 A100:80, H100:90）
   ```

3. **编译**
   ```bash
   make -j$(nproc)
   ```

## 运行测试

编译完成后，在 `build/` 目录下执行对应可执行文件：
```bash
# 运行端到端模型推理测试（默认编译）
./modelTest

# 若开启了其他测试：
./linearTest          # 线性层功能验证
./nonlinearTest       # 非线性函数近似验证
./benchmarkTest       # 性能基准测试
./SwinModelTest       # Swin Transformer 专用测试
```

## 注意事项

- **CUDA 架构适配**: 请根据实际 GPU 型号修改 `CMAKE_CUDA_ARCHITECTURES`（如 RTX 3090 用 86，A100 用 80）
- **HEonGPU 依赖**: 确保 HEonGPU 1.1+ 已正确安装，且 CMake 能通过 `find_package(HEonGPU 1.1 CONFIG)` 找到
- **测试数据说明**: `data/` 目录下 CSV 文件为真实推理流程中的输入/权重快照，用于验证密态推理准确率

---

*如需扩展模型支持或反馈问题，欢迎提交 Issue 或 PR。*