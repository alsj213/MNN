# MNN 论文-代码对照技术报告

> 论文: *"MNN: A Universal and Efficient Inference Engine"* (MLSys 2020, Alibaba Group)
> ArXiv: https://arxiv.org/abs/2002.12418

本文档将 MNN 论文中提出的主要技术创新点与代码实现进行一一对照，标注精确的文件路径和关键行号。

---

## 目录

1. [Winograd 卷积优化](#1-winograd-卷积优化)
2. [ARM NEON/汇编级内核优化](#2-arm-neon汇编级内核优化)
3. [算子融合（Op Fusion）](#3-算子融合op-fusion)
4. [内存优化——两级 BufferAllocator](#4-内存优化两级-bufferallocator)
5. [自动代码生成（codegen）](#5-自动代码生成codegen)
6. [INT8 量化推理](#6-int8-量化推理)
7. [GPU 后端与自动调优](#7-gpu-后端与自动调优)
8. [异构后端调度](#8-异构后端调度)
9. [低内存模式（4-bit / 混合精度权重量化）](#9-低内存模式4-bit--混合精度权重量化)
10. [Shape 推理系统](#10-shape-推理系统)
11. [Strassen 矩阵乘法](#11-strassen-矩阵乘法)
12. [NC4HW4 内存布局与 Packing](#12-nc4hw4-内存布局与-packing)

---

## 1. Winograd 卷积优化

### 论文创新点

论文提出使用 Winograd 变换减少卷积中的乘法次数，将 F(m×m, k×k) 卷积的乘法复杂度从 O(m²·k²) 降低到 O((m+k-1)²)。MNN 实现了 F(2,3)、F(4,3)、F(6,3)、F(8,3) 四种变体，并针对不同平台做了深度优化。

### 代码实现

#### 数学基础——Winograd 矩阵生成

| 文件 | 说明 |
|------|------|
| `source/math/WingoradGenerater.hpp:16-44` | `WinogradGenerater` 类定义，构造三个变换矩阵 A（输出变换）、B（输入变换）、G（权重变换） |
| `source/math/WingoradGenerater.cpp:17-80` | 核心数学函数：`computeF()`（对角分母）、`computeT()`（T 矩阵）、`computeL()`（Lagrange 多项式系数）、`computeB()`（B = L·T） |
| `source/math/WingoradGenerater.cpp:94-135` | `computeA()`（输出求值矩阵）、`computeFDiag()`（对角缩放） |
| `source/math/WingoradGenerater.cpp:139-217` | 构造函数：接受 `computeUnit`（tile 大小）、`kernelSize`、`interp`（插值参数，默认 0.5） |

#### 运行时 Winograd 调度

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/compute/ConvolutionWinogradImpl.cpp:39-50` | `canUseWinograd()`：门控条件——要求方形 kernel > 1×1、dilation == 1、stride == 1 |
| `source/backend/cpu/compute/ConvolutionFloatFactory.cpp:173-182` | 工厂中 Winograd 路径选择：检查 `winogradMemoryUsed` hint → `bestWinogradUnit()` → `createWinogradImpl()` |

#### CPU Winograd 执行——Pack 变体

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/compute/ConvolutionPackWinograd.cpp:28-80` | Pack 模式 Winograd（ARM/SSE/AVX2，pack=4/8），构造函数中设置源/目标变换函数 |
| `source/backend/cpu/compute/ConvolutionPackWinograd.cpp:142-214` | `bestWinogradUnit()`：遍历候选 tile 大小，计算 `originCost / winogradCost` 减少率，选择最优 |
| `source/backend/cpu/compute/ConvolutionPackFreeWinograd.cpp:29-299+` | Pack-Free 模式（AVX-512，pack=16），三阶段流水线：Source Transform → Batched GEMM → Dest Transform |

#### 优化的变换函数

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/compute/WinogradOptFunction.cpp:27` | `_sourceTransformUnit4x4Pack12`：4×4 tile 源变换，使用 `Vec4`（4 宽 SIMD 抽象） |
| `source/backend/cpu/compute/WinogradOptFunction.cpp:280` | `_sourceTransformUnit8x8Pack12`：8×8 tile，F(8,3) 系数如 `s00*36 - s20*49 + s40*14 - s60` |
| `source/backend/cpu/compute/WinogradOptFunction.cpp:1464` | `chooseSourceTransform()`：按 k=4/6/8 选择源变换 |
| `source/backend/cpu/compute/WinogradOptFunction.cpp:1495` | `chooseDestTransform()`：按 k=4/6/8 + h=2..7 选择目标变换 |

#### 平台特定 Winograd 实现

| 平台 | 文件 |
|------|------|
| ARM FP16 | `source/backend/arm82/Arm82WinogradOptFunc.cpp` |
| ARM64 汇编 | `source/backend/cpu/arm/arm64/MNNConvDwF23MulTransUnit.S`、`MNNConvDwF23SourceTransUnit.S` |
| x86 AVX | `source/backend/cpu/x86_x64/avx/WinogradFunctions.cpp` |
| x86 AVX-512 | `source/backend/cpu/x86_x64/avx512/WinogradFunctions.cpp` |
| OpenCL buffer | `source/backend/opencl/execution/buffer/ConvBufWinograd.cpp` |
| OpenCL image | `source/backend/opencl/execution/image/ConvWinograd.cpp` |
| Metal | `source/backend/metal/MetalConvolutionWinograd.mm` + `shader/MetalConvolutionWinograd.metal` |
| Vulkan | `source/backend/vulkan/image/execution/VulkanConvolutionWinograd.cpp` |
| CUDA | `source/backend/cuda/execution/ConvWinogradExecution.cu` + `WinogradTrans.cuh` |
| INT8 Winograd | `source/backend/cpu/compute/ConvInt8Winograd.cpp` |

---

## 2. ARM NEON/汇编级内核优化

### 论文创新点

论文强调 MNN 对移动 ARM CPU 的深度优化，包括手写 NEON 汇编 GEMM 微内核、针对不同 ARM 架构版本（ARMv8.0/8.2/8.6/SME2）的特化实现。

### 代码实现

#### ARM64 汇编内核（`source/backend/cpu/arm/arm64/`，100+ 文件）

| 类别 | 文件 | 说明 |
|------|------|------|
| INT8 GEMM 基础 | `MNNGemmInt8AddBiasScale_16x4_Unit.S:72` | 16×4 tiled INT8 GEMM，融合 bias/scale/ReLU，使用 NEON `fmla.4s`/`fmul.4s` |
| INT8 GEMM ARMv8.2 | `MNNGemmInt8AddBiasScale_ARMV82_Unit.S` | 使用 `sdot` 指令（ARMv8.2 点积）加速 INT8 矩阵乘 |
| INT8 GEMM ARMv8.6 | `MNNGemmInt8AddBiasScale_ARMV86_Unit.S` | 使用 ARMv8.6 i8mm（矩阵乘）指令 |
| INT8 GEMM 快速路径 | `MNNGemmInt8AddBiasScale_16x4_Unit_FAST.S` | 快速路径变体 |
| SME2 INT8 GEMM | `sme2_asm/MNNGemmInt8AddBiasScale16x32_SME2_w8_Fp32.S` + 9 more | Scalable Matrix Extension 2 内核 |
| FP32 GEMM | `MNNPackedMatMul.S` | FP32 矩阵乘 |
| Depthwise Conv | `MNNConvRunForLineDepthwise.S` | Depthwise 卷积 |
| INT8 Depthwise | `MNNLineDepthWiseInt8AddBiasScaleUnit.S` | INT8 Depthwise 卷积 |
| 图像格式转换 | `MNNC3ToC4Fast.S`、`MNNNV21ToBGRUnit.S` 等 | NEON 加速的颜色空间转换 |

#### INT8 GEMM 内核内部结构

`MNNGemmInt8AddBiasScale_16x4_Unit.S:16-82`：
- `MLA_WEIGHTZERO`（line 16）：融合乘加补偿权重零点
- `ReLU_FP32_4/3/2/1`（lines 19-46）：使用 `fmin`/`fmax` 的向量化 ReLU
- `MUL_SCALE4/3/2/1`（lines 47-64）：向量化逐通道 scale 乘法
- 主函数处理 16 输出行 × 4 输出列/迭代

#### FP16 汇编（ARM82，`source/backend/arm82/asm/arm64/`，30+ 文件）

包括 FP16 GEMM、Winograd 变换、depthwise conv、gelu、exp、pack/unpack、quantize 操作。

#### 运行时 CPU 特性检测

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/compute/CommonOptFunction.cpp` | ARM 特性检测：SME2、I8MM、SDot、BF16 |
| `source/backend/cpu/compute/Int8FunctionsOpt.cpp:21-60` | 声明 extern 汇编函数，按 CPU 能力分发（ARMv8.0 → ARMv8.2 sdot → ARMv8.6 i8mm → SME2） |
| `source/backend/cpu/x86_x64/FunctionDispatcher.cpp:41-86` | x86 运行时分发：`MNNFunctionInit()` 检测 SSE4.1/AVX2/FMA3，渐进覆盖函数指针 |

---

## 3. 算子融合（Op Fusion）

### 论文创新点

论文提出在图级别将多个算子融合为一个执行单元，减少内存访问和 kernel launch 开销。MNN 实现了 Conv+BN+ReLU、Attention、LayerNorm、GeLU 等多种融合模式。

### 代码实现

#### 转换器级图融合（模板匹配）

**目录：** `tools/converter/source/optimizer/merge/`（20+ 融合 pass）

| 融合 Pass | 文件 | 说明 |
|-----------|------|------|
| Attention | `FuseAttention.cpp:18-80+` | 匹配 Q·K^T → Softmax → ·V 模式，含 GQA 检测（lines 27-41） |
| LayerNorm | `FuseLayerNorm.cpp:44-60+` | 匹配 Mean→Sub→Square→Mean→Add(eps)→RSqrt→Mul→Add(gamma/beta) |
| LayerNorm RMS | `FuseLayerNormRMS.cpp` | RMSNorm 变体 |
| GeLU | `FuseGeLu.cpp` | 匹配 tanh(0.7978845608·x·(1+0.044715·x²)) 模式 |
| SplitGeLU | `FuseSplitGeLu.cpp` | Split + GeLU 融合算子 |
| Conv+BN+ReLU | `ConvBNReluFuseToConvInt8.cpp` | 融合卷积、批归一化、ReLU 为 INT8 conv |
| Conv+Dilate | `ConvDilateFuse.cpp` | 将膨胀融合进卷积参数 |
| GroupNorm | `FuseGroupNorm.cpp` | 组归一化融合 |
| FMHA v2 | `FuseFmhaV2.cpp` | 融合多头注意力 v2 |
| FMHCA | `FuseFmhca.cpp` | 融合多头交叉注意力 |

所有融合 pass 使用 `TemplateMerge` 模式匹配框架（`tools/converter/source/optimizer/TemplateMerge.hpp`）。

#### 运行时级融合

| 文件 | 说明 |
|------|------|
| `codegen/OpFuse.cpp:65-100+` | `mergeConvolutionAndPrelu()`：将 Conv + PReLU 融合为 `ExtraConvolution2DPrelu` |
| `codegen/OpFuse.cpp` | `opFuse()`：识别可融合的逐元素算子链 |

#### Transformer 专用融合门控

| 文件 | 说明 |
|------|------|
| `source/shape/SizeComputer.hpp:176-187` | `REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE` 宏，`MNN_SUPPORT_TRANSFORMER_FUSE` 编译开关 |
| `source/shape/ShapeRegister.cpp:241-248` | 在 `MNN_SUPPORT_TRANSFORMER_FUSE` 下注册 SplitGeLU、FmhaV2、Fmhca、Attention、LinearAttention 的 ShapeComputer |

---

## 4. 内存优化——两级 BufferAllocator

### 论文创新点

论文提出针对移动设备的内存优化策略，包括内存池管理、in-place 操作、延迟分配等，最小化内存分配开销。

### 代码实现

**核心文件：** `source/core/BufferAllocator.hpp:24-243`、`source/core/BufferAllocator.cpp:1-698`

#### 第一级：EagerBufferAllocator（即时分配器）

`source/core/BufferAllocator.hpp:113-193`

- **Free-list + split/merge 语义**
- `alloc()`（`.cpp:201`）：先搜 `mCurrentFreeList`（线程本地组），再搜全局 `mFreeList`，用 `lower_bound` 做 best-fit。若 `allocSize > minAllocSize`，分割分配（lines 244-264）
- `free()`（`.cpp:306`）：释放节点回 freelist
- `returnMemory()`（`.cpp:274`）：父节点 `useCount` 归零时合并子节点（lines 283-303）
- **多线程屏障**（lines 350-372）：`barrierBegin()`/`barrierEnd()` 同步组 freelist

#### 第二级：DeferBufferAllocator（延迟分配器）

`source/core/BufferAllocator.hpp:205-242`

- **离线最优偏移计算**——一次遍历后分配
- `compute()`（`.cpp:573`）：遍历链表，分配偏移，调用 `apply()`
- `apply()`（`.cpp:596`）：分配单个连续 buffer，回填所有张量指针
- `fuse_to_left()`（`.cpp:633`）：融合相邻空闲节点实现最优打包

#### 后端分配器

| 文件 | 说明 |
|------|------|
| `source/core/BufferAllocator.cpp:50` | `DefaultAllocator`：原始 `MNNMemoryAllocAlign` |
| `source/core/BufferAllocator.cpp:66` | `MmapAllocator`：大权重的内存映射文件分配器 |
| `source/core/BufferAllocator.cpp:161` | `RecurseAllocator`：委托给父级 `BufferAllocator` |

---

## 5. 自动代码生成（codegen）

### 论文创新点

论文提出通过自动代码生成减少手写 kernel 的工程量，同时保持高性能。MNN 的 codegen 系统针对逐元素算子链生成融合 kernel。

### 代码实现

**目录：** `codegen/`

#### 核心框架

| 文件 | 说明 |
|------|------|
| `codegen/SourceModule.hpp:57-80` | `SourceModule` 类：接受 `Target` 后端，从融合 `Node` 子图构建 kernel 代码 |
| `codegen/SourceModule.cpp:91-162` | `buildKernel()`：遍历拓扑排序的算子节点，通过 `Target` 接口发射 load/compute/store |
| `codegen/OpFuse.cpp:65-100+` | 图分析：识别可融合子图（逐元素链），调用 `mergeConvolutionAndPrelu()` |

#### Target 后端

| Target | 文件 | 说明 |
|--------|------|------|
| OpenCL | `codegen/opencl/OpenCLTarget.hpp:1-28` | 生成 `__kernel` 函数，`FLOAT4` 向量类型 |
| Metal | `codegen/metal/MetalTarget.hpp:1-28` | 生成 Metal Shading Language，`M4` (float4) 类型 |
| CUDA | `codegen/cuda/CUDATarget.hpp` + `CUDATarget.cpp` | CUDA kernel 生成（最大实现，43KB） |
| C/CPU | `codegen/cpu/c/SourceTargetCodeGen.cpp:1-219` | 生成 C 代码，AST 中间表示：`ForExpr`、`BinaryExpr`、`AssignExpr` |
| CPU Plugin | `codegen/cpu/CPUPluginModule.cpp:40-80` | 生成可加载的 CPU 插件模块 |

#### AST 中间表示

`codegen/cpu/AST.hpp:29-222`：完整 AST——`NumberExpr`、`BinaryExpr`、`UnaryExpr`、`ForExpr`、`IfExpr`、`LoopExpr` 等，通过 visitor 模式调用 `codegen(SourceTarget*)`。

---

## 6. INT8 量化推理

### 论文创新点

论文提出 INT8 量化推理以加速移动设备上的模型执行，包括权重量化和激活量化，以及融合的量化 GEMM 内核。

### 代码实现

#### INT8 卷积执行器

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/compute/ConvInt8TiledExecutor.hpp:18` | `ConvInt8TiledExecutor` 基类：`packWeightAndQuantInfo()`、`reorderWeight()`、`initializeConvInt8QuantInfo()` |
| `source/backend/cpu/compute/ConvInt8TiledExecutor.cpp:51` | `DenseConvInt8TiledExecutor`：动态量化支持 `mQuantFunc`、逐通道求和 `mSumByAxisLFunc` |
| `source/backend/cpu/compute/Int8FunctionsOpt.cpp:2447-2518` | 函数注册：`Int8GemmKernelFast`、ARM82 FP16 输出、ARMV82/86 特化路径 |

#### INT8 汇编内核（ARM64）

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/arm/arm64/MNNGemmInt8AddBiasScale_16x4_Unit.S` | 16×4 基础 INT8 GEMM |
| `source/backend/cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV82_Unit.S` | sdot GEMM |
| `source/backend/cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV86_Unit.S` | i8mm GEMM |
| `source/backend/cpu/arm/arm64/MNNPackC4Int8ForMatMulA_ARM82.S` | INT8 输入 packing |
| `source/backend/cpu/arm/arm64/MNNSumWeightInt8Arm82.S` / `Arm86.S` | 权重求和（反量化补偿） |
| `source/backend/cpu/arm/arm64/sme2_asm/` | 10 个 SME2 INT8 GEMM 变体 |

#### INT8 支撑算子

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/CPUBinaryInt8.cpp/.hpp` | INT8 二元运算 |
| `source/backend/cpu/CPUDepthwiseConvInt8.cpp/.hpp` | INT8 Depthwise 卷积 |
| `source/backend/cpu/CPUFloatToInt8.cpp/.hpp` | Float→INT8 量化 |
| `source/backend/cpu/CPUInt8ToFloat.cpp/.hpp` | INT8→Float 反量化 |
| `source/backend/cpu/CPUPoolInt8.cpp/.hpp` | INT8 池化 |
| `source/backend/cpu/compute/ConvInt8Winograd.cpp` | INT8 Winograd 卷积 |

---

## 7. GPU 后端与自动调优

### 论文创新点

论文提出跨平台 GPU 加速（OpenCL/Metal/Vulkan）以及针对不同硬件的自动调优机制。

### 代码实现

#### OpenCL 后端

| 目录/文件 | 说明 |
|-----------|------|
| `source/backend/opencl/execution/cl/` | 70+ `.cl` kernel 文件：conv_2d、gemm、winograd、matmul、depthwise、attention 等 |
| `source/backend/opencl/core/OpenCLGemmTune.cpp:35` | `isCandidateValid()`：验证 GEMM 配置候选（local memory、workgroup 大小、精度） |
| `source/backend/opencl/core/OpenCLGemmTune.cpp:131` | `GemmlocalWSTune()`：按问题规模搜索已调优参数数据库 |
| `source/backend/opencl/core/OpenCLGemmTune.cpp:181` | `getGemmParams()`：主入口——查缓存 → tuneLws → 全搜索 |
| `source/backend/opencl/core/runtime/OpenCLRuntime.hpp:59-65` | `TuneInfo` 结构体：存储 program name、MD5、global/local sizes、耗时 |
| `source/backend/opencl/core/runtime/OpenCLRuntime.hpp:156-160` | `tunedGemmParamsMap()`/`tunedLwsMap()`：按问题规模缓存的 GEMM 参数 |

#### Metal 后端

`source/backend/metal/`：32 个 `.metal` shader 文件，包括 `MetalConvolutionWinograd.metal`、`MetalConvolutionGEMM.metal`、`MetalMatMul.metal` 等。

#### Vulkan 后端

`source/backend/vulkan/`：70+ compute shader（`.comp`），包括 `attention_fused.comp`、`attention_prefill_kblock_*.comp`、`attention_decode_q1_*.comp`、`attention_kvcache_update.comp` 等 Transformer 专用 kernel。

#### CUDA 后端

`source/backend/cuda/execution/`：包括 `ConvWinogradExecution.cu`、`WinogradTrans.cuh` 等。

---

## 8. 异构后端调度

### 论文创新点

论文提出自动的异构后端选择机制，根据设备能力自动选择最优执行后端。

### 代码实现

| 文件 | 说明 |
|------|------|
| `source/core/Backend.cpp:24-30` | 全局注册表：`map<MNNForwardType, pair<RuntimeCreator*, bool>>` |
| `source/core/Backend.cpp:55-84` | `registerBackend()`：`std::call_once` 注册 CPU + 条件注册 CoreML/NNAPI/QNN/OpenCL/Metal/NeuroPilot |
| `source/core/Backend.cpp:86-104` | `MNNGetExtraRuntimeCreator()`：按 forward type 查询 `RuntimeCreator` |
| `source/core/Schedule.cpp:111-160` | `getAppropriateType()`：**优先级自动选择**——HIAI > CoreML > TensorRT > CUDA > OpenCL > Metal > Vulkan > CPU |
| `source/core/Schedule.cpp:140-155` | OpenCL 低功耗验证：`Power_Low` 时检查 `STATUS_SUPPORT_POWER_LOW`，不支持则 fallback |
| `source/core/Schedule.cpp:291-436` | `Schedule::schedule()`：为每个 config 调用 `getAppropriateType()` → `_scheduleUnit()` → `generateScheduleGraph()` |
| `source/core/Schedule.cpp:162-280` | `generateScheduleGraph()`：支持基于 tensor 和基于 op 的子图提取，前向/后向传播 |

---

## 9. 低内存模式（4-bit / 混合精度权重量化）

### 论文创新点

（MNN 2.x 扩展）支持 4-bit/8-bit 混合精度权重量化，结合动态激活量化，在移动设备上以极低内存运行大型模型。

### 代码实现

#### 入口与门控

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/compute/ConvolutionFloatFactory.cpp:47` | `#ifdef MNN_LOW_MEMORY` 保护整个低内存代码路径 |
| `source/backend/cpu/compute/ConvolutionFloatFactory.cpp:153-160` | `Memory_Low` 模式 → `DenseConvInt8TiledExecutor`（动态量化）；否则 → `DenseConvolutionTiledExecutor`（量化权重反量化） |

#### 动态量化函数（NEON 内在函数）

`source/backend/cpu/arm/arm64/low_memory/MNNDynamicQuantFunctions.hpp:1-688`：

- `MNNAsyLocalQuantInfo_EP10/12/16_FP32()`：非对称量化，加载 min/max → 计算 `qscale=255/diff`、`qbias=-(255*min/diff+128)` → 用 `vclezq_f32` 位掩码处理零范围块

#### 低内存汇编内核（`source/backend/cpu/arm/arm64/low_memory/`，14 文件）

| 文件 | 说明 |
|------|------|
| `MNNGemmInt8AddBiasScale_ARMV82_w4_Unit.S` | INT4 权重 GEMM + ARMv8.2 sdot |
| `MNNGemmInt8AddBiasScale_ARMV86_w4_Unit.S` | INT4 权重 GEMM + ARMv8.6 i8mm |
| `MNNGemmInt8AddBiasScale_16x4_w4_Unit.S` | INT4 权重 GEMM 基础路径 |
| `MNNDynamicQuantFP32_Pack4.S` / `Pack8.S` | 动态 FP32→INT8 量化 |
| `MNNDynamicQuantAndReorder_ARM82.S` | 量化 + 重排 for GEMM |
| `MNNGeneralIm2col_Fp32Arm82.S` / `Arm86.S` / `Sme2.S` | 低内存 im2col |

#### 核心函数注册

`source/backend/cpu/compute/CommonOptFunction.cpp:4743-4773`：
```cpp
#ifdef MNN_LOW_MEMORY
    gCoreFunction->MNNAbsMax = MNNAbsMaxFP32;           // abs max for 量化
    gCoreFunction->MNNDynamicQuant = MNNDynamicQuantFP32; // 对称批量量化
    gCoreFunction->MNNAsyQuantFunc = MNNAsyQuantFunc;     // 非对称批量量化
#endif
```

#### 权重重排

`source/backend/cpu/compute/CommonOptFunction.cpp:639`：`MNNReorderWeightInt4()` 将 4-bit 打包权重重排为 GEMM 内核期望的布局。

---

## 10. Shape 推理系统

### 论文创新点

论文提出独立的 shape 推理系统，在执行前计算输出张量形状，支持动态 shape。

### 代码实现

| 文件 | 说明 |
|------|------|
| `source/shape/SizeComputer.hpp:24` | `SizeComputer` 基类：`onComputeSize()`（纯虚）、`onComputeFlops()`、`needInputContent()` |
| `source/shape/SizeComputer.hpp:78` | `SizeComputerSuite` 单例注册表：按 `OpType` 数组索引，O(1) 查找 |
| `source/shape/SizeComputer.hpp:134-187` | 注册宏：`REGISTER_SHAPE`、`REGISTER_SHAPE_INPUTS`、`REGISTER_SHAPE_INPUTS_TRANSFORMER_FUSE` |
| `source/shape/ShapeRegister.cpp:126-249` | `registerShapeOps()`：注册 ~110 个 ShapeComputer，覆盖所有支持的算子 |
| `source/shape/ShapeConvolution.cpp:25` | 卷积 shape：从 kernel/stride/dilation/pad 计算输出 H/W |
| `source/shape/ShapeMatMul.cpp:29` | MatMul shape：处理 transpose 标志、broadcast 维度 |

**总计：** `source/shape/` 下约 77 个文件，每个算子类型一个。

---

## 11. Strassen 矩阵乘法

### 论文创新点

论文提到对 1×1 卷积使用 Strassen 算法减少乘法次数。

### 代码实现

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/compute/Convolution1x1Strassen.cpp` | 1×1 卷积的 Strassen 优化实现 |
| `source/backend/cpu/compute/StrassenMatmulComputor.hpp` | Strassen 矩阵乘法封装 |
| `source/backend/cpu/compute/ConvolutionFloatFactory.cpp:121-183` | 工厂中检查 `is1x1Conv` → 选择 `Convolution1x1Strassen` |

---

## 12. NC4HW4 内存布局与 Packing

### 论文创新点

论文提出 NC4HW4（channels packed by 4）内存布局，利用 SIMD 向量化加速计算。

### 代码实现

#### 内部数据格式

| 文件 | 说明 |
|------|------|
| `source/core/Tensor.cpp:20-45` | 三种维度格式：`CAFFE`(NCHW)、`TENSORFLOW`(NHWC)、`CAFFE_C4`(NC4HW4) |
| `schema/default/Tensor.fbs` | `MNN_DATA_FORMAT_NC4HW4` 定义 |

#### Packing/Unpacking 函数

| 文件 | 说明 |
|------|------|
| `source/backend/cpu/compute/CommonOptFunction.h:61-98` | 声明：`MNNPackC4`、`MNNUnpackC4`、`MNNPackTranspose`、`MNNUnpackTranspose`、`MNNTranspose32Bit`、`MNNPackC4ForMatMul_A`、`MNNPackForMatMul_B` |
| `source/backend/cpu/x86_x64/sse/ReorderFunctions.cpp:79-180` | SSE 实现：`MNNUnpackC4()` 用 `_MM_TRANSPOSE4_PS` 从 NC4HW4 转 planar，`MNNPackC4()` 反向 |
| `source/backend/cpu/arm/arm64/MNNPackC4.S` | ARM64 NEON 汇编 C4 packing |
| `source/backend/cpu/x86_x64/sse/PackedFunction.cpp` | GEMM 专用 packing：`_SSE_MNNPackC4ForMatMul_A`、`_SSE_MNNPackForMatMul_B` |

---

## 附录：核心数据流总览

```
模型加载 (FlatBuffers .mnn)
  │
  ├─ Interpreter::createFromFile() → Net*
  │
  └─ createSession()
       │
       ├─ Schedule::schedule()
       │    ├─ getAppropriateType() → 自动选择后端 [创新点 8]
       │    ├─ generateScheduleGraph() → 子图提取
       │    └─ initPipelineInfosFromOps() → OpCacheInfo[]
       │
       └─ new Session()
            ├─ createPipelineBackend() → 主后端 + CPU 备份
            └─ new Pipeline()
                 │
                 ├─ encode()
                 │    ├─ SizeComputer → shape 推理 [创新点 10]
                 │    └─ GeometryComputer → 几何分解
                 │
                 ├─ allocMemory()
                 │    ├─ BufferAllocator → 两级内存管理 [创新点 4]
                 │    └─ _createExecutions() → 后端 Execution
                 │
                 └─ execute()
                      └─ cmd.execution->onExecute()
                           │
                           ├─ Conv: Winograd [创新点 1] / im2col+GEMM / Strassen [创新点 11]
                           ├─ GEMM: NEON asm [创新点 2] / INT8 [创新点 6] / INT4 [创新点 9]
                           ├─ Elementwise: codegen fused kernel [创新点 5]
                           └─ GPU: OpenCL/Metal/Vulkan + auto-tune [创新点 7]
```

---

## 附录：论文创新点与代码位置速查表

| # | 论文创新点 | 核心代码路径 |
|---|-----------|-------------|
| 1 | Winograd 卷积 | `source/math/WingoradGenerater.*` + `source/backend/cpu/compute/Convolution*Winograd*` |
| 2 | ARM NEON 汇编优化 | `source/backend/cpu/arm/arm64/*.S`（100+ 文件）+ `source/backend/arm82/asm/` |
| 3 | 算子融合 | `tools/converter/source/optimizer/merge/Fuse*.cpp` + `codegen/OpFuse.cpp` |
| 4 | 内存优化 | `source/core/BufferAllocator.*`（两级分配器） |
| 5 | 自动代码生成 | `codegen/SourceModule.*` + `codegen/{opencl,metal,cuda,cpu}/` |
| 6 | INT8 量化推理 | `source/backend/cpu/compute/ConvInt8TiledExecutor.*` + `Int8FunctionsOpt.*` |
| 7 | GPU 后端 + 自动调优 | `source/backend/opencl/core/OpenCLGemmTune.cpp` + 各 GPU 后端 kernel |
| 8 | 异构后端调度 | `source/core/Schedule.cpp:111`（优先级选择）+ `source/core/Backend.cpp:55`（注册） |
| 9 | 低内存 4-bit 量化 | `source/backend/cpu/arm/arm64/low_memory/` + `CommonOptFunction.cpp:4743` |
| 10 | Shape 推理系统 | `source/shape/SizeComputer.*` + `source/shape/ShapeRegister.cpp` |
| 11 | Strassen 矩阵乘 | `source/backend/cpu/compute/Convolution1x1Strassen.cpp` |
| 12 | NC4HW4 内存布局 | `source/core/Tensor.cpp` + `CommonOptFunction.h:61`（Packing 函数） |
