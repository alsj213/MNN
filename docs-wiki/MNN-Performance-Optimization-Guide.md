# MNN 性能优化指南

本文档详细说明 MNN 的性能优化技术和最佳实践。

## 目录
1. [性能优化概述](#性能优化概述)
2. [图优化](#图优化)
3. [内存优化](#内存优化)
4. [计算优化](#计算优化)
5. [量化优化](#量化优化)
6. [LLM 优化](#llm-优化)
7. [性能分析工具](#性能分析工具)
8. [平台特定优化](#平台特定优化)

## 性能优化概述

### 优化层次

```
模型层优化（剪枝、蒸馏）
    ↓
图优化（算子融合、常量折叠）
    ↓
内存优化（内存复用、NC4HW4）
    ↓
计算优化（SIMD、多线程、量化）
    ↓
硬件优化（GPU、NPU）
```

### 性能指标

| 指标 | 说明 | 目标 |
|------|------|------|
| **延迟 (Latency)** | 单次推理时间 | 越低越好 |
| **吞吐量 (Throughput)** | 单位时间处理样本数 | 越高越好 |
| **内存占用** | 运行时内存使用 | 越低越好 |
| **模型大小** | 模型文件大小 | 越小越好 |
| **功耗** | 能量消耗 | 越低越好 |

## 图优化

### 1. 算子融合

#### Conv + BN + ReLU 融合

```
融合前: Conv2D -> BatchNorm -> ReLU
融合后: Conv2D (with fused BN and ReLU)
```

**原理**：
- BN 可以融合到 Conv 的权重和偏置中
- ReLU 可以作为 Conv 的激活函数

**效果**：
- 减少内存访问
- 减少算子调度开销
- 提升 30-50% 性能

#### 启用方法

在模型转换时自动启用：

```bash
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode biz
```

### 2. 常量折叠

```
优化前: x = input * 2.0; y = x + 1.0
优化后: y = constant_value (如果 input 是常量)
```

**效果**：
- 减少运行时计算
- 减少内存占用

### 3. Transformer 融合

MNN 支持 Transformer 特定的融合优化：

```
融合前: Q = Linear(x); K = Linear(x); V = Linear(x)
       Attention = Softmax(Q @ K.T / sqrt(d)) @ V
融合后: Attention = FusedMultiHeadAttention(x)
```

**启用方法**：

```bash
cmake .. -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
```

## 内存优化

### 1. NC4HW4 格式

MNN 内部使用 NC4HW4 格式优化 SIMD 性能：

```
原始 NCHW: [N, C, H, W]
NC4HW4:    [N, C/4, H, W, 4]
```

**优势**：
- 通道按 4 打包，利于 SIMD 指令（NEON/SSE）
- 减少内存访问次数
- 提高缓存命中率

**性能提升**：20-40%

### 2. 内存复用

#### 动态内存分配

```cpp
// 配置动态内存复用
ScheduleConfig config;
config.type = MNN_FORWARD_CPU;
// 默认启用动态内存复用
```

**原理**：
- 算子执行完后释放内存到池中
- 后续算子从池中复用内存
- 减少内存分配次数

**效果**：
- 减少 50-70% 内存占用
- 提升 10-20% 性能

#### 内存对齐

```cpp
// 使用对齐的内存分配
float* buffer = (float*)MNNMemoryAllocAlign(
    size * sizeof(float), 
    MNN_MEMORY_ALIGN_DEFAULT  // 通常是 64 字节
);
```

**效果**：
- 提升缓存性能
- 支持 SIMD 指令

### 3. 低内存模式

```bash
# 编译时启用
cmake .. -DMNN_LOW_MEMORY=ON
```

**特性**：
- 权重量化
- 延迟加载
- 内存映射

**效果**：
- 减少 50-75% 内存占用
- 略微降低性能（5-10%）

## 计算优化

### 1. SIMD 优化

#### NEON (ARM)

```cpp
#ifdef MNN_USE_NEON
#include <arm_neon.h>

void vectorAdd(const float* a, const float* b, float* c, int size) {
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    // 处理剩余元素
    for (; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}
#endif
```

**性能提升**：4-8x

#### SSE/AVX (x86)

```cpp
#ifdef MNN_USE_SSE
#include <immintrin.h>

void vectorMul(const float* a, const float* b, float* c, int size) {
    int i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(c + i, vc);
    }
    // 处理剩余元素
    for (; i < size; ++i) {
        c[i] = a[i] * b[i];
    }
}
#endif
```

**性能提升**：8-16x

### 2. 多线程并行

#### 配置线程数

```cpp
ScheduleConfig config;
config.type = MNN_FORWARD_CPU;
config.numThread = 4;  // 使用 4 个线程

auto session = interpreter->createSession(config);
```

**最佳实践**：
- 移动端：2-4 线程
- 服务器：物理核心数
- 避免超线程

#### 并行模式

```cpp
// MNN 提供的并行宏
MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
    // 并行处理 batch 维度
    for (int b = tId; b < batch; b += threadNumber) {
        processData(b);
    }
} MNN_CONCURRENCY_END();
```

**性能提升**：接近线性（理想情况）

### 3. Winograd 卷积

Winograd 算法减少卷积计算量：

```cpp
// 配置 Winograd
interpreter->setSessionHint(
    Interpreter::WINOGRAD_MEMORY_LEVEL, 
    3  // 0-3，越大使用越多候选单元
);
```

**适用场景**：
- 3x3 卷积
- stride = 1
- 输入输出通道数较大

**性能提升**：
- 计算量减少 2.25x
- 实际性能提升 1.5-2x

## 量化优化

### 1. 量化类型对比

| 类型 | 精度 | 性能提升 | 模型大小 |
|------|------|---------|---------|
| **FP32** | 基准 | 1x | 100% |
| **FP16** | 略降 | 1.5-2x | 50% |
| **INT8** | 轻微降 | 2-4x | 25% |
| **INT4** | 可接受 | 3-6x | 12.5% |

### 2. 训练后量化 (PTQ)

```bash
# 使用 MNN 量化工具
python tools/quantization/quantize.py \
    --mnn_model model.mnn \
    --quant_imgs calibration_data/ \
    --quant_bits 8 \
    --output model_quant.mnn
```

**步骤**：
1. 准备校准数据集（100-1000 张图片）
2. 运行量化工具
3. 验证精度

### 3. 混合精度

关键层使用 FP32，其他层使用 INT8，在模型转换时指定。

## LLM 优化

### 1. KVCache 管理

#### 启用 KVCache

```cpp
Llm* llm = Llm::createLLM(config_path);

// KVCache 自动管理
llm->response("你好", &std::cout);
```

**优化**：
- Prefill 阶段：计算所有 token 的 KV
- Decode 阶段：只计算新 token 的 KV
- 复用历史 KV

**性能提升**：10-100x（Decode 阶段）

#### KVCache 持久化

```cpp
// 配置 KVCache 持久化
interpreter->setSessionHint(
    Interpreter::KVCACHE_SIZE_LIMIT,
    1024 * 1024 * 100  // 100MB
);
```

**效果**：
- 跨会话复用 KVCache
- 减少重复计算

### 2. 推测解码

#### EAGLE 推测解码

```json
{
  "llm_config": "config.json",
  "llm_weight": "model.mnn",
  "draft_config": "draft_config.json",
  "draft_weight": "draft.mnn"
}
```

**原理**：
- 使用小模型生成候选 tokens
- 大模型并行验证
- 接受正确的 tokens

**性能提升**：1.5-3x

### 3. 量化

#### 4-bit 量化

```bash
python llmexport.py \
    --path /path/to/model \
    --export mnn \
    --quant_bit 4 \
    --quant_block 128 \
    --dst_path ./MODEL
```

**效果**：
- 模型大小减少 75%
- 内存占用减少 75%
- 速度提升 1.5-2x
- 精度损失 < 1%

### 4. 注意力优化

#### Flash Attention

```cpp
// 启用 Flash Attention
interpreter->setSessionHint(
    Interpreter::ATTENTION_OPTION,
    8  // 启用 Flash Attention
);
```

**效果**：
- 减少内存占用
- 提升计算速度
- 支持更长序列

## 性能分析工具

### 1. 内置性能分析

```cpp
// 获取内存使用
float memory = 0.0f;
interpreter->getSessionInfo(session, Interpreter::MEMORY, &memory);
MNN_PRINT("Memory: %.2f MB\n", memory);

// 获取 FLOPS
float flops = 0.0f;
interpreter->getSessionInfo(session, Interpreter::FLOPS, &flops);
MNN_PRINT("FLOPS: %.2f M\n", flops);
```

### 2. 算子级性能分析

```cpp
// 使用回调获取每个算子的性能
auto before = [](const std::vector<Tensor*>& inputs, const OperatorInfo* info) {
    MNN_PRINT("Before: %s\n", info->name().c_str());
    return true;
};

auto after = [](const std::vector<Tensor*>& outputs, const OperatorInfo* info) {
    MNN_PRINT("After: %s, FLOPS: %.2f M\n", 
              info->name().c_str(), info->flops());
    return true;
};

interpreter->runSessionWithCallBackInfo(session, before, after);
```

### 3. 性能基准测试

```bash
# 运行基准测试
./benchmark.out \
    --model model.mnn \
    --backend CPU \
    --thread 4 \
    --loop 100
```

## 平台特定优化

### 1. iOS / macOS (Metal)

```cpp
ScheduleConfig config;
config.type = MNN_FORWARD_METAL;

BackendConfig metalConfig;
metalConfig.precision = BackendConfig::Precision_Low;  // FP16
config.backendConfig = &metalConfig;
```

**性能提升**：2-5x（相比 CPU）

### 2. Android (OpenCL/Vulkan)

```cpp
// OpenCL
ScheduleConfig config;
config.type = MNN_FORWARD_OPENCL;

// Vulkan
config.type = MNN_FORWARD_VULKAN;
```

### 3. NVIDIA GPU (CUDA)

```cpp
ScheduleConfig config;
config.type = MNN_FORWARD_CUDA;

BackendConfig cudaConfig;
cudaConfig.precision = BackendConfig::Precision_Low;  // FP16
config.backendConfig = &cudaConfig;
```

**性能提升**：5-20x（相比 CPU）

### 4. ARM CPU

#### 大小核调度

```cpp
interpreter->setSessionHint(
    Interpreter::CPU_LITTLECORE_DECREASE_RATE,
    50  // 小核性能为大核的 50%
);
```

#### ARM 扩展指令

```bash
# 启用 ARM82 (FP16)
cmake .. -DMNN_ARM82=ON

# 启用 SME2
cmake .. -DMNN_SME2=ON
```

**性能提升**：
- ARM82: 1.5-2x
- SME2: 2-3x

## 性能优化检查清单

### 模型层面
- [ ] 使用模型压缩（剪枝、蒸馏）
- [ ] 选择合适的模型架构
- [ ] 减少模型参数量

### 转换层面
- [ ] 启用算子融合
- [ ] 启用 Transformer 融合
- [ ] 使用量化

### 运行时层面
- [ ] 选择合适的后端
- [ ] 配置合适的线程数
- [ ] 启用低内存模式（如需要）
- [ ] 使用 FP16（GPU）

### 代码层面
- [ ] 减少 Host-Device 数据传输
- [ ] 批处理操作
- [ ] 复用内存
- [ ] 使用 SIMD 指令

### LLM 特定
- [ ] 启用 KVCache
- [ ] 使用 4-bit 量化
- [ ] 启用 Flash Attention
- [ ] 考虑推测解码

## 性能优化案例

### 案例 1: ResNet50 优化

**初始性能**：
- CPU (4 threads): 100 ms
- 内存: 200 MB

**优化步骤**：
1. 启用算子融合 → 70 ms
2. 使用 INT8 量化 → 35 ms
3. 启用 Winograd → 25 ms
4. 优化线程数 → 20 ms

**最终性能**：
- CPU (4 threads): 20 ms (5x 提升)
- 内存: 50 MB (4x 减少)

### 案例 2: LLM 优化

**初始性能**：
- Prefill (512 tokens): 2000 ms
- Decode (per token): 100 ms
- 内存: 8 GB

**优化步骤**：
1. 启用 KVCache → Decode: 50 ms
2. 4-bit 量化 → 内存: 2 GB
3. Flash Attention → Prefill: 1500 ms
4. EAGLE 推测解码 → Decode: 25 ms (等效)

**最终性能**：
- Prefill (512 tokens): 1500 ms (1.3x 提升)
- Decode (per token): 25 ms (4x 提升，等效)
- 内存: 2 GB (4x 减少)

## 参考资源

- [MNN 官方文档](https://www.yuque.com/mnn/cn)
- [性能测试工具](../../benchmark/)
- [量化工具](../../tools/quantization/)
- [LLM 优化](../../transformers/llm/)
- [架构文档](./MNN_architecture.md)
- [ARM CPU 优化 Skill](../../skills/arm-cpu-optimize/SKILL.md)
