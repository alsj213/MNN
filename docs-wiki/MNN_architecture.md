# MNN 仓库实现分析

MNN 是阿里巴巴开源的轻量级深度学习**推理引擎**，支持 CNN / Transformer / LLM / Diffusion 模型，目标平台涵盖移动端和服务器。以下从架构、核心流程、后端、LLM 子系统等维度进行分析。

---

## 一、整体架构

```
用户代码
  │
  ├─ Session API (低层) ─── Interpreter → Session → Pipeline
  │     操作 Tensor，适合固定 shape、极致性能场景
  │
  └─ Module API (高层, 推荐) ─── Module::load → onForward(VARP)
        Express 动态图，LLM/Diffusion/大多数现代工作负载使用
```

模型文件格式为 **FlatBuffers**（`.mnn`），通过 `schema/default/*.fbs` 定义 ~200 个算子。

---

## 二、核心推理流水线（Interpreter → Session → Pipeline）

| 组件 | 文件 | 职责 |
|------|------|------|
| **Interpreter** | `source/core/Interpreter.cpp` | 顶层入口，持有模型 buffer 和 FlatBuffers Net，管理 Session |
| **Schedule** | `source/core/Schedule.cpp` | 从模型图构建执行计划，自动选择后端（优先级：HIAI > CoreML > TensorRT > CUDA > OpenCL > Metal > Vulkan > CPU） |
| **Session** | `source/core/Session.cpp` | 持有一个或多个 Pipeline（按后端分区），管理 resize/run/clone |
| **Pipeline** | `source/core/Pipeline.cpp` | 单个后端上的算子执行计划：encode（shape 推理 + 几何分解）→ allocMemory（创建 Execution + 分配内存）→ execute |
| **Tensor** | `source/core/Tensor.cpp` | 数据容器，支持 NCHW / NHWC / NC4HW4（SIMD packed）格式 |

关键数据流：
```
createFromFile() → FlatBuffers Net
  → createSession() → Schedule::schedule() → 构建 OpCacheInfo[]
    → new Session() → 创建 Pipeline（主后端 + CPU 备份）
      → resize() → encode() + allocMemory()
      → run() → pipeline.execute() → cmd.execution->onExecute()
```

---

## 三、后端架构（Backend）

三层抽象：**Runtime**（工厂）→ **Backend**（内存管理 + Execution 创建）→ **Execution**（单算子执行）

| 后端 | 目录 | Forward Type | 说明 |
|------|------|-------------|------|
| CPU | `source/backend/cpu/` | 0 | 始终可用，Compiler_Loop |
| Metal | `source/backend/metal/` | 1 | iOS/macOS GPU |
| CUDA | `source/backend/cuda/` | 2 | NVIDIA GPU |
| OpenCL | `source/backend/opencl/` | 3 | 跨平台 GPU |
| Vulkan | `source/backend/vulkan/` | 7 | 跨平台 GPU |
| ARM82 | `source/backend/arm82/` | 13 | ARM FP16 扩展 |
| QNN / HiAI / TensorRT / NeuroPilot / CoreML | 各自目录 | USER_* | NPU / 推理加速器 |
| MUSA | `source/backend/musa/` | 15 | 摩尔线程 GPU |

**Op 分发机制**：每个后端的 `onCreate()` 返回 `Execution*` 或 `nullptr`，后者触发 fallback 到 CPU 备份后端。

---

## 四、Express API（Module / VARP）

高层 PyTorch 风格接口：

- **VARP**：智能指针 → `Variable` → `Expr`（计算图节点）
- **Executor**：单例，管理 Runtime 并执行表达式图
- **Module**：抽象基类，`PipelineModule` 是核心实现（从 `.mnn` 加载，反序列化为子 Module 链）
- **Functional API**：`_Const()`, `_Input()`, `_Conv()`, `_MatMul()`, `_Softmax()` 等

---

## 五、LLM 子系统

### 5.1 Python 导出（`transformers/llm/export/`）

```
HuggingFace 模型
  → LlmModel.from_pretrained()
  → (可选) AWQ / SmoothQuant / OmniQuant 量化
  → FakeLinear 卸载权重（节省 GPU 显存）
  → torch.onnx.export()
  → MNNConverter → llm.mnn + llm.mnn.weight
```

`ModelMapper` 是核心映射注册表，支持 **30+ 架构**（llama, qwen, qwen2, qwen3, qwen3_moe, chatglm, gemma2/3/4, phi, internlm, baichuan, deepseek-vl 等），默认 fallback 到 LlamaForCausalLM。

### 5.2 C++ 推理引擎（`transformers/llm/engine/`）

```
Llm (基类) ──→ Omni (多模态: vision + audio + talker)
         └──→ Embedding (句子嵌入)
         └──→ Talker (语音生成)
```

**关键流程**：
1. `Llm::createLLM(config)` → 加载 `LlmConfig` JSON → `Module::load()` + `Tokenizer` + `Sampler`
2. **Prefill**：一次性 forward 所有输入 token
3. **Decode**：逐 token forward → `Sampler::sample(logits)` → 采样下一个 token
4. **停止条件**：EOS 或达到 max tokens

### 5.3 KVCache 管理

`KVMeta` 结构体通过 hint 指针传递给 MNN 运行时：
- `add` / `remove` / `reserve` 信号驱动缓存增删
- 支持 **磁盘持久化**（`mmap`）实现跨会话 prefix cache
- 支持 **对话编辑**（`eraseHistory()` 任意擦除 KV 范围）
- **KV 共享**：gemma4 等模型在层间共享 KV 缓存

### 5.4 采样策略

流水线模式，可组合步骤：`Penalty → TopK → TopP → MinP → TFS → Typical → Temperature → Select`

支持：Greedy、Temperature、Top-K、Top-P (Nucleus)、Min-P、TFS、Typical、N-gram Penalty、Logit Bias、Banned Tokens。

### 5.5 推测解码（Speculative Decoding）

| 策略 | 类 | 说明 |
|------|-----|------|
| 自回归 | `ArGeneration` | 标准逐 token 生成 |
| Lookahead | `LookaheadGeneration` | N-gram 前瞻推测 |
| MTP | `MtpGeneration` | 多 token 预测 |
| EAGLE | `EagleGeneration` | 草稿模型 + 树注意力 + 验证 |

---

## 六、Diffusion 支持（`transformers/diffusion/`）

| 模型 | 架构 |
|------|------|
| Stable Diffusion 1.5 | CLIP + UNet + VAE |
| Taiyi Chinese SD | CLIP-like + UNet + VAE |
| Sana | Qwen3-0.6B 编码器 + DiT + Flow Matching |
| SD 3.5 | 三重文本编码器 + Transformer + Flow Matching |

---

## 七、模型转换器（`tools/converter/`）

支持 **7 种源格式**：TensorFlow、Caffe、ONNX、MNN（自转换）、TFLite、PyTorch、JSON。

不只是格式转换——包含图优化 pass（算子融合、冗余消除、Transformer 注意力模式融合），输出 FlatBuffers `.mnn` 格式。

---

## 八、量化支持

**通用 PTQ**（`tools/quantization/`）：KL 散度、ADMM、EMA、Moving Average、MAX_ABS

**LLM 专用**（`transformers/llm/export/utils/`）：
- **AWQ**：激活感知权重量化
- **SmoothQuant**：将量化难度从激活迁移到权重
- **OmniQuant**：可学习量化参数
- **HQQ**：半二次量化

配置：`quant_bit`（4/8）、`quant_block`（分组大小，默认 64）、`sym`（对称/非对称）、`embed_bit`、`act_bit` 等。

---

## 九、Schema / Op 定义（`schema/default/`）

### 主 Schema（`MNN.fbs`）

**OpType 枚举**：定义 ~200 个算子类型，包括：
- 标准 DNN：`Convolution`, `Pooling`, `BatchToSpaceND`, `InnerProduct`, `Softmax`, `ReLU`, `Sigmoid`, `MatMul`, `LayerNorm`
- Transformer 专用：`Attention` (299), `FmhaV2` (300), `Fmhca` (301), `LinearAttention` (305), `GroupNorm` (304), `SplitGeLU` (303), `MoE` (58)
- 量化：`ConvInt8` (513), `Int8ToFloat` (514), `FloatToInt8` (517)
- 控制流：`While` (600), `If` (601)
- 内部：`Raster` (128), `ConvertTensor` (129), `Plugin` (256), `Extra` (512)

**`AttentionParam`**：
```
table AttentionParam {
    kv_cache: bool = true;
    kv_shared_layer: string;
    layer_index: int = -1;
    kv_shared_layer_index: int = -1;
}
```

**`LoopParam`**：用于几何分解算子，定义基于循环的执行（`RegionCommand`）。

---

## 十、Shape 推理与几何分解

### SizeComputer（`source/shape/`）

抽象类，计算输出张量形状。通过 `REGISTER_SHAPE` 宏注册到 `SizeComputerSuite` 全局注册表。约 50+ 文件，如 `ShapeConvolution.cpp`, `ShapeMatMul.cpp`, `ShapeAttention.cpp`。

### GeometryComputer（`source/geometry/`）

将复杂算子分解为简单算子（供不原生支持的后端使用）。约 40+ 文件，如 `GeometryConv2D.cpp`, `GeometryBatchMatMul.cpp`。

编译器类型：
- `Compiler_Geometry`（默认）：分解复杂算子为原语
- `Compiler_Origin`：直接传递算子给后端（GPU 后端）
- `Compiler_Loop`：基于循环执行（CPU 后端）

---

## 十一、构建系统（`CMakeLists.txt`）

### 关键构建选项

| 选项 | 默认值 | 用途 |
|------|--------|------|
| `MNN_BUILD_LLM` | OFF | LLM 推理库 |
| `MNN_BUILD_LLM_OMNI` | OFF | LLM + 视觉/音频 |
| `MNN_BUILD_DIFFUSION` | OFF | Diffusion 模型支持 |
| `MNN_LOW_MEMORY` | OFF | 低内存权重量化 |
| `MNN_SUPPORT_TRANSFORMER_FUSE` | OFF | 融合 Transformer 算子 |
| `MNN_BUILD_CONVERTER` | OFF | 模型转换工具 |
| `MNN_BUILD_MINI` | OFF | 最小构建（无几何分解，精简算子） |
| `MNN_SME2` | ON | ARM SME2 指令 |
| `MNN_SUPPORT_BF16` | OFF | BFloat16 算子 |

### 依赖链

- `MNN_BUILD_LLM` → 强制 `MNN_LOW_MEMORY=ON`, `MNN_SUPPORT_TRANSFORMER_FUSE=ON`
- `MNN_BUILD_LLM_OMNI` → 额外强制 `MNN_BUILD_OPENCV=ON`, `MNN_BUILD_AUDIO=ON`
- `MNN_BUILD_DIFFUSION` → 强制 `MNN_LOW_MEMORY=ON`, `MNN_SUPPORT_TRANSFORMER_FUSE=ON`, `MNN_BUILD_OPENCV=ON`
- `MNN_BUILD_MINI` → 强制 `MNN_SKIPBUILD_GEOMETRY=ON`, `MNN_REDUCE_SIZE=ON`

---

## 十二、测试基础设施（`test/`）

自定义测试框架（非 Google Test）：`MNNTestSuite.h` → `MNNTestCase` 基类 + `MNNTestSuiteRegister` 宏注册。

| 目录 | 数量 | 焦点 |
|------|------|------|
| `test/op/` | 93 文件 | 算子正确性 |
| `test/core/` | 11 文件 | Backend, Tensor, Memory, ThreadPool |
| `test/expr/` | 20 文件 | Express API |
| `test/speed/` | 13 文件 | 性能基准 |
| `test/grad/` | 14 文件 | 梯度计算 |
| `test/model/` | 3 文件 | 端到端模型测试 |

---

## 十三、关键设计亮点

1. **内存高效导出**：`FakeLinear` 技巧——ONNX 导出时用零权重占位符替换 Linear 层，再通过 `OnnxRebuilder` 回注真实权重，使消费级 GPU 也能导出 70B+ 模型。

2. **KVCache 作为运行时原语**：LLM 引擎只发出 add/remove/reserve 信号，实际张量内存管理委托给 MNN 核心运行时，可利用后端特定优化（如 OpenCL buffer 复用）。

3. **异构后端调度**：单个模型可按算子粒度分配到不同后端（如 CPU+FP32 做 vision 编码，GPU+FP16 做 LLM backbone），通过 `Pipeline` 间的 `Copy` 算子桥接。

4. **图编译器式转换器**：Converter 不仅做格式转换，还包含图级优化 pass，融合 Transformer 注意力模式，对执行效率有显著提升。

5. **可组合采样流水线**：Sampler 采用 pipeline 模式，各采样步骤可灵活组合，支持复杂的采样策略（如 N-gram penalty + Top-P + Temperature）。

---

## 十四、核心数据流总览

```
用户代码
  │
  ├─ Interpreter API ─── createFromFile() ─── FlatBuffers Net
  │     │
  │     └─ createSession() ─── Schedule::schedule() ─── 构建 op 列表
  │           │                    │
  │           │                    ├─ getAppropriateType() ─── 自动选择后端
  │           │                    └─ initPipelineInfosFromOps() ─── OpCacheInfo[]
  │           │
  │           └─ new Session() ─── 为每个 pipeline:
  │                 │                   ├─ createPipelineBackend() ─── 主后端 + CPU 备份
  │                 │                   └─ new Pipeline()
  │                 │
  │                 ├─ resize() ─── encode() ─── shape 推理 + 几何分解
  │                 │            └── allocMemory() ─── 创建 Execution + 分配张量内存
  │                 │
  │                 └─ run() ─── pipeline.execute() ─── cmd.execution->onExecute()
  │
  └─ Express API ─── VARP / EXPRP 计算图
        │
        ├─ Executor::getGlobalExecutor() ─── 管理运行时
        │
        └─ Module::load() ─── PipelineModule ─── StaticModule(s)
              │
              └─ module->forward(inputs) ─── 内部使用 Interpreter/Session

LLM 层:
  Llm::createLLM(config) ─── load() ─── Module::load() + Tokenizer + Sampler
       │
       ├─ forward(input_ids) ─── embedding → 模型 forward → logits
       │
       └─ generate() ─── prefill 循环 → decode 循环（sample → forward → 重复）
                │
                └─ 生成策略: Ar / Lookahead / MTP / Eagle
```
