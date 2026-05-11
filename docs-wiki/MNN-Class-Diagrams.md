# MNN 核心类图

本文档使用 Mermaid 图表展示 MNN 核心类的结构和关系。

## 1. 核心推理类图

```mermaid
classDiagram
    class Tensor {
        -halide_buffer_t mBuffer
        -InsideDescribe* mDescribe
        +DimensionType getDimensionType()
        +halide_type_t getType()
        +host() T*
        +deviceId() uint64_t
        +shape() vector~int~
        +copyFromHostTensor(hostTensor) bool
        +copyToHostTensor(hostTensor) bool
        +map(MapType, DimensionType) void*
        +unmap(MapType, DimensionType, ptr) void
    }

    class Interpreter {
        -Content* mNet
        +createFromFile(file)$ Interpreter*
        +createFromBuffer(buffer, size)$ Interpreter*
        +createSession(config) Session*
        +runSession(session) ErrorCode
        +resizeSession(session) void
        +getSessionInput(session, name) Tensor*
        +getSessionOutput(session, name) Tensor*
        +releaseSession(session) bool
        +setSessionMode(mode) void
        +setCacheFile(file, keySize) void
    }

    class Session {
        -vector~Pipeline*~ pipelines
        -map~string,Tensor*~ inputs
        -map~string,Tensor*~ outputs
        +resize() ErrorCode
        +run() ErrorCode
        +getInput(name) Tensor*
        +getOutput(name) Tensor*
    }

    class Backend {
        <<abstract>>
        -MNNForwardType mType
        +onCreate(inputs, outputs, op)* Execution*
        +onResizeBegin() void
        +onResizeEnd()* ErrorCode
        +onExecuteBegin()* void
        +onExecuteEnd()* void
        +onAcquire(tensor, storageType)* MemObj*
        +onClearBuffer()* bool
        +onCopyBuffer(src, dst)* void
        +type() MNNForwardType
    }

    class Runtime {
        <<abstract>>
        -RuntimeHint mHint
        +onCreate(config, origin)* Backend*
        +onReset(numThread, config, full) void
        +onGabageCollect(level)* void
        +onGetMemoryInMB() float
        +onSetCache(buffer, size) bool
        +onGetCache() pair
        +onMeasure(inputs, outputs, op, info) bool
    }

    class Execution {
        <<abstract>>
        -Backend* mBackend
        +onResize(inputs, outputs)* ErrorCode
        +onExecute(inputs, outputs)* ErrorCode
        +backend() Backend*
    }

    class Pipeline {
        -Backend* mBackend
        -vector~Command~ mCommands
        +encode() ErrorCode
        +allocMemory() ErrorCode
        +execute() ErrorCode
    }

    class RuntimeCreator {
        <<abstract>>
        +onCreate(info)* Runtime*
        +onValid(info) bool
    }

    Interpreter "1" --> "*" Session : manages
    Session "1" --> "*" Pipeline : contains
    Pipeline "1" --> "1" Backend : uses
    Backend "1" --> "*" Execution : creates
    Runtime "1" --> "*" Backend : creates
    RuntimeCreator "1" --> "*" Runtime : creates
    Pipeline "*" --> "*" Tensor : operates on
    Execution "*" --> "*" Tensor : processes
```

## 2. Express API 类图

```mermaid
classDiagram
    class VARP {
        -Variable* mContent
        +operator->() Variable*
        +get() Variable*
        +fix(type) VARP
        +sum(axis) VARP
        +mean(axis) VARP
    }

    class Variable {
        -Expr* mFrom
        -vector~int~ mShape
        -halide_type_t mType
        +expr() Expr*
        +shape() vector~int~
        +getInfo() Info*
        +resize(dims) void
        +writeMap() void*
        +readMap() void*
    }

    class Expr {
        <<abstract>>
        -Op* mOp
        -vector~VARP~ mInputs
        -string mName
        +get() Op*
        +inputs() vector~VARP~
        +name() string
        +outputSize() int
    }

    class Executor {
        -map~MNNForwardType,Runtime*~ mRuntimes
        -vector~ComputeCache~ mCaches
        +getGlobalExecutor()$ Executor*
        +onCreate(inputs, op) Expr*
        +runCache(cache) ErrorCode
        +setGlobalExecutorConfig(type, config, numThread) void
        +getAttr(key) string
        +setAttr(key, value) void
    }

    class Module {
        <<abstract>>
        -vector~VARP~ mParameters
        -vector~Module*~ mChildren
        -string mName
        -string mType
        +onForward(inputs)* vector~VARP~
        +forward(input) VARP
        +parameters() vector~VARP~
        +loadParameters(params) bool
        +load(inputs, outputs, buffer, length, config)$ Module*
        +clone(module, shareParams)$ Module*
    }

    class PipelineModule {
        -Interpreter* mInterpreter
        -Session* mSession
        -map~string,VARP~ mInputs
        -map~string,VARP~ mOutputs
        +onForward(inputs) vector~VARP~
        +load(inputs, outputs, buffer, length, config)$ Module*
    }

    VARP "1" --> "1" Variable : wraps
    Variable "1" --> "1" Expr : created by
    Expr "*" --> "*" VARP : has inputs
    Executor "1" --> "*" Expr : executes
    Module <|-- PipelineModule : implements
    PipelineModule "1" --> "1" Interpreter : uses
    PipelineModule "1" --> "1" Session : uses
```

## 3. LLM 子系统类图

```mermaid
classDiagram
    class Llm {
        -LlmContext* mContext
        -KVMeta* mMeta
        -LlmConfig* mConfig
        -Tokenizer* mTokenizer
        -Sampler* mSampler
        -Module* mModule
        -Executor* mExecutor
        +createLLM(config_path)$ Llm*
        +load() bool
        +forward(input_ids, is_prefill) VARP
        +sample(logits, offset, size) int
        +response(input_ids, os, end_with, max_tokens) void
        +generate(input_ids, max_tokens) vector~int~
        +tokenizer_encode(query) vector~int~
        +tokenizer_decode(token) string
        +reset() void
        +switchMode(stage) void
    }

    class Embedding {
        +createEmbedding(config_path, load)$ Embedding*
        +load() bool
        +ids_embedding(ids) VARP
        +txt_embedding(txt) VARP
        +dist(var0, var1)$ float
        +cos_sim(var0, var1)$ float
        +dim() int
    }

    class LlmConfig {
        +string model_dir
        +string tokenizer_file
        +int max_new_tokens
        +float temperature
        +int top_k
        +float top_p
        +string backend_type
        +int thread_num
        +bool kvcache_mmap
        +int quant_bit
        +int quant_block
    }

    class Tokenizer {
        -sentencepiece::SentencePieceProcessor* mProcessor
        -Tiktoken* mTiktoken
        +encode(text) vector~int~
        +decode(id) string
        +is_stop(id) bool
        +load(filename) bool
    }

    class Sampler {
        -vector~SamplerStep*~ mSteps
        -float mTemperature
        -int mTopK
        -float mTopP
        +sample(logits, token_id) int
        +addStep(step) void
        +reset() void
    }

    class Generation {
        <<abstract>>
        -Llm* mLlm
        +generate(input_ids, max_tokens)* vector~int~
        +reset()* void
    }

    class ArGeneration {
        +generate(input_ids, max_tokens) vector~int~
    }

    class LookaheadGeneration {
        -int mWindowSize
        -int mNgramSize
        +generate(input_ids, max_tokens) vector~int~
    }

    class EagleGeneration {
        -Llm* mDraftModel
        -int mTreeSize
        +generate(input_ids, max_tokens) vector~int~
    }

    class KVMeta {
        +size_t add
        +size_t remove
        +int* reserve
        +int n_reserve
    }

    class LlmContext {
        +int prompt_len
        +int gen_seq_len
        +int all_seq_len
        +vector~int~ history_tokens
        +vector~int~ output_tokens
        +string generate_str
        +LlmStatus status
        +int64_t prefill_us
        +int64_t decode_us
    }

    Llm <|-- Embedding : extends
    Llm "1" --> "1" LlmConfig : uses
    Llm "1" --> "1" Tokenizer : uses
    Llm "1" --> "1" Sampler : uses
    Llm "1" --> "1" Module : uses
    Llm "1" --> "1" KVMeta : manages
    Llm "1" --> "1" LlmContext : has
    Llm "1" --> "1" Generation : uses
    Generation <|-- ArGeneration : implements
    Generation <|-- LookaheadGeneration : implements
    Generation <|-- EagleGeneration : implements
```

## 4. 后端实现类图

```mermaid
classDiagram
    class Backend {
        <<abstract>>
        +onCreate(inputs, outputs, op)* Execution*
        +onAcquire(tensor, storageType)* MemObj*
        +onClearBuffer()* bool
    }

    class CPUBackend {
        -BufferAllocator* mStaticAllocator
        -BufferAllocator* mDynamicAllocator
        +onCreate(inputs, outputs, op) Execution*
        +onAcquire(tensor, storageType) MemObj*
        +onClearBuffer() bool
        +onCopyBuffer(src, dst) void
    }

    class MetalBackend {
        -id~MTLDevice~ mDevice
        -id~MTLCommandQueue~ mQueue
        -BufferAllocator* mBufferPool
        +onCreate(inputs, outputs, op) Execution*
        +onAcquire(tensor, storageType) MemObj*
        +onClearBuffer() bool
    }

    class CUDABackend {
        -CUDARuntime* mCUDARuntime
        -BufferAllocator* mBufferPool
        +onCreate(inputs, outputs, op) Execution*
        +onAcquire(tensor, storageType) MemObj*
        +onClearBuffer() bool
    }

    class OpenCLBackend {
        -OpenCLRuntime* mOpenCLRuntime
        -cl::Context mContext
        -cl::CommandQueue mQueue
        +onCreate(inputs, outputs, op) Execution*
        +onAcquire(tensor, storageType) MemObj*
        +onClearBuffer() bool
    }

    class VulkanBackend {
        -VulkanRuntime* mRuntime
        -VkDevice mDevice
        -VkQueue mQueue
        +onCreate(inputs, outputs, op) Execution*
        +onAcquire(tensor, storageType) MemObj*
        +onClearBuffer() bool
    }

    class Execution {
        <<abstract>>
        +onResize(inputs, outputs)* ErrorCode
        +onExecute(inputs, outputs)* ErrorCode
    }

    class CPUConvolution {
        -float* mWeight
        -float* mBias
        -ConvolutionCommon* mCommon
        +onResize(inputs, outputs) ErrorCode
        +onExecute(inputs, outputs) ErrorCode
    }

    class MetalConvolution {
        -id~MTLBuffer~ mWeight
        -id~MTLBuffer~ mBias
        -id~MTLComputePipelineState~ mPipeline
        +onResize(inputs, outputs) ErrorCode
        +onExecute(inputs, outputs) ErrorCode
    }

    Backend <|-- CPUBackend : implements
    Backend <|-- MetalBackend : implements
    Backend <|-- CUDABackend : implements
    Backend <|-- OpenCLBackend : implements
    Backend <|-- VulkanBackend : implements
    
    Execution <|-- CPUConvolution : implements
    Execution <|-- MetalConvolution : implements
    
    CPUBackend --> CPUConvolution : creates
    MetalBackend --> MetalConvolution : creates
```

## 类图说明

### 1. 核心推理类图
展示了 MNN 的核心推理架构，包括：
- **Interpreter**: 模型加载和会话管理的入口
- **Session**: 推理会话，管理多个 Pipeline
- **Pipeline**: 单个后端的执行流水线
- **Backend/Runtime**: 硬件后端抽象
- **Execution**: 单个算子的执行器
- **Tensor**: 数据容器

### 2. Express API 类图
展示了高级动态图 API：
- **VARP**: 智能指针，包装 Variable
- **Variable**: 变量，持有 Expr
- **Expr**: 表达式，表示计算图节点
- **Executor**: 全局执行器
- **Module**: 模型抽象，PipelineModule 是主要实现

### 3. LLM 子系统类图
展示了 LLM 推理引擎：
- **Llm**: 基类，提供核心推理能力
- **Embedding**: 句子嵌入特化
- **Tokenizer**: 分词器
- **Sampler**: 采样器
- **Generation**: 生成策略（AR/Lookahead/EAGLE）
- **KVMeta/LlmContext**: 运行时状态

### 4. 后端实现类图
展示了不同硬件后端的实现：
- 各后端继承 Backend 抽象类
- 每个后端创建对应的 Execution 实现
- 统一的接口，不同的实现
