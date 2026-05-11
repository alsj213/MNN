# MNN 推理流程时序图

本文档使用 Mermaid 时序图展示 MNN 的关键执行流程。

## 1. Session API 推理流程

### 1.1 模型加载和会话创建

```mermaid
sequenceDiagram
    participant User
    participant Interpreter
    participant Schedule
    participant Session
    participant Pipeline
    participant Backend
    participant Runtime

    User->>Interpreter: createFromFile(model.mnn)
    Interpreter->>Interpreter: 加载 FlatBuffers 模型
    Interpreter-->>User: Interpreter*

    User->>Interpreter: createSession(config)
    Interpreter->>Schedule: schedule(net, config)
    Schedule->>Schedule: getAppropriateType()<br/>(自动选择后端)
    Schedule->>Schedule: initPipelineInfosFromOps()<br/>(构建算子列表)
    Schedule-->>Interpreter: OpCacheInfo[]

    Interpreter->>Session: new Session(pipelines)
    Session->>Runtime: onCreate(config)
    Runtime-->>Session: Backend*
    Session->>Pipeline: new Pipeline(backend, ops)
    Pipeline-->>Session: Pipeline*
    Session-->>Interpreter: Session*
    Interpreter-->>User: Session*
```

### 1.2 Resize 和内存分配

```mermaid
sequenceDiagram
    participant User
    participant Interpreter
    participant Session
    participant Pipeline
    participant Backend
    participant SizeComputer
    participant GeometryComputer

    User->>Interpreter: resizeTensor(input, dims)
    Interpreter->>Interpreter: 更新输入张量形状

    User->>Interpreter: resizeSession(session)
    Interpreter->>Session: resize()
    
    loop 每个 Pipeline
        Session->>Pipeline: encode()
        Pipeline->>SizeComputer: onComputeSize(op, inputs, outputs)
        SizeComputer-->>Pipeline: 输出形状
        
        opt 需要几何分解
            Pipeline->>GeometryComputer: onCompute(op, inputs, outputs)
            GeometryComputer-->>Pipeline: 分解后的算子
        end
        
        Pipeline-->>Session: 完成形状推导
        
        Session->>Pipeline: allocMemory()
        Pipeline->>Backend: onCreate(inputs, outputs, op)
        Backend-->>Pipeline: Execution*
        Pipeline->>Backend: onAcquire(tensor, STATIC)
        Backend-->>Pipeline: MemObj*
        Pipeline-->>Session: 完成内存分配
    end
    
    Session-->>Interpreter: 完成
    Interpreter-->>User: 完成
```

### 1.3 推理执行

```mermaid
sequenceDiagram
    participant User
    participant Interpreter
    participant Session
    participant Pipeline
    participant Backend
    participant Execution
    participant Tensor

    User->>Interpreter: getSessionInput(session, "input")
    Interpreter->>Session: getInput("input")
    Session-->>Interpreter: Tensor*
    Interpreter-->>User: Tensor*

    User->>Tensor: copyFromHostTensor(hostTensor)
    Tensor->>Tensor: 拷贝数据到设备

    User->>Interpreter: runSession(session)
    Interpreter->>Session: run()
    
    loop 每个 Pipeline
        Session->>Pipeline: execute()
        Pipeline->>Backend: onExecuteBegin()
        
        loop 每个 Command
            Pipeline->>Execution: onExecute(inputs, outputs)
            Execution->>Execution: 执行算子计算
            Execution-->>Pipeline: ErrorCode
        end
        
        Pipeline->>Backend: onExecuteEnd()
        Pipeline-->>Session: ErrorCode
    end
    
    Session-->>Interpreter: ErrorCode
    Interpreter-->>User: ErrorCode

    User->>Interpreter: getSessionOutput(session, "output")
    Interpreter->>Session: getOutput("output")
    Session-->>Interpreter: Tensor*
    Interpreter-->>User: Tensor*

    User->>Tensor: copyToHostTensor(hostTensor)
    Tensor->>Tensor: 拷贝数据到主机
```

## 2. Module API 推理流程

### 2.1 模型加载

```mermaid
sequenceDiagram
    participant User
    participant Module
    participant PipelineModule
    participant Interpreter
    participant Session
    participant Executor

    User->>Module: load(inputs, outputs, buffer, config)
    Module->>Executor: getGlobalExecutor()
    Executor-->>Module: Executor*
    
    Module->>PipelineModule: new PipelineModule()
    PipelineModule->>Interpreter: createFromBuffer(buffer)
    Interpreter-->>PipelineModule: Interpreter*
    
    PipelineModule->>Interpreter: createSession(config)
    Interpreter-->>PipelineModule: Session*
    
    PipelineModule->>PipelineModule: 创建输入/输出 VARP
    PipelineModule-->>Module: Module*
    Module-->>User: Module*
```

### 2.2 前向传播

```mermaid
sequenceDiagram
    participant User
    participant Module
    participant PipelineModule
    participant Interpreter
    participant Session
    participant VARP
    participant Tensor

    User->>Module: forward(input_varp)
    Module->>PipelineModule: onForward([input_varp])
    
    PipelineModule->>VARP: input_varp->writeMap()
    VARP->>Tensor: 获取写入指针
    Tensor-->>VARP: void*
    VARP-->>PipelineModule: void*
    
    PipelineModule->>PipelineModule: 拷贝数据到输入张量
    
    PipelineModule->>Interpreter: resizeSession(session)
    Interpreter->>Session: resize()
    Session-->>Interpreter: 完成
    Interpreter-->>PipelineModule: 完成
    
    PipelineModule->>Interpreter: runSession(session)
    Interpreter->>Session: run()
    Session-->>Interpreter: ErrorCode
    Interpreter-->>PipelineModule: ErrorCode
    
    PipelineModule->>VARP: output_varp->readMap()
    VARP->>Tensor: 获取读取指针
    Tensor-->>VARP: void*
    VARP-->>PipelineModule: void*
    
    PipelineModule-->>Module: [output_varp]
    Module-->>User: output_varp
```

## 3. LLM 推理流程

### 3.1 LLM 初始化

```mermaid
sequenceDiagram
    participant User
    participant Llm
    participant LlmConfig
    participant Module
    participant Tokenizer
    participant Sampler

    User->>Llm: createLLM(config_path)
    Llm->>LlmConfig: load(config_path)
    LlmConfig-->>Llm: LlmConfig*
    
    Llm->>Llm: load()
    Llm->>Module: load(inputs, outputs, model_file, config)
    Module-->>Llm: Module*
    
    Llm->>Tokenizer: load(tokenizer_file)
    Tokenizer-->>Llm: Tokenizer*
    
    Llm->>Sampler: new Sampler(config)
    Sampler-->>Llm: Sampler*
    
    Llm->>Llm: initRuntime()
    Llm-->>User: Llm*
```

### 3.2 文本生成流程（Prefill + Decode）

```mermaid
sequenceDiagram
    participant User
    participant Llm
    participant Tokenizer
    participant Module
    participant Sampler
    participant KVMeta

    User->>Llm: response("你好", os)
    Llm->>Llm: apply_chat_template("你好")
    Llm->>Tokenizer: encode(prompt)
    Tokenizer-->>Llm: input_ids[]
    
    Note over Llm: === Prefill 阶段 ===
    Llm->>Llm: switchMode(Prefill)
    Llm->>KVMeta: add = prompt_len
    Llm->>Llm: embedding(input_ids)
    Llm->>Module: forward(input_embeds)
    Module-->>Llm: logits
    
    Llm->>Sampler: sample(logits)
    Sampler-->>Llm: next_token
    Llm->>Tokenizer: decode(next_token)
    Tokenizer-->>Llm: text
    Llm->>User: 输出 text
    
    Note over Llm: === Decode 阶段 ===
    Llm->>Llm: switchMode(Decode)
    
    loop 直到 EOS 或达到 max_tokens
        Llm->>KVMeta: add = 1
        Llm->>Llm: embedding([next_token])
        Llm->>Module: forward(input_embeds)
        Module-->>Llm: logits
        
        Llm->>Sampler: sample(logits)
        Sampler-->>Llm: next_token
        
        opt 是停止词
            Llm->>Llm: break
        end
        
        Llm->>Tokenizer: decode(next_token)
        Tokenizer-->>Llm: text
        Llm->>User: 输出 text
    end
    
    Llm-->>User: 完成生成
```

### 3.3 推测解码流程（EAGLE）

```mermaid
sequenceDiagram
    participant Llm as 主模型
    participant Eagle as EAGLE生成器
    participant Draft as 草稿模型
    participant Sampler

    Llm->>Eagle: generate(input_ids, max_tokens)
    
    loop 直到达到 max_tokens
        Note over Eagle: === 草稿阶段 ===
        Eagle->>Draft: forward(current_token)
        Draft-->>Eagle: draft_logits
        Eagle->>Sampler: sample(draft_logits)
        Sampler-->>Eagle: draft_tokens[tree_size]
        
        Note over Eagle: === 验证阶段 ===
        Eagle->>Llm: forward([current_token] + draft_tokens)
        Llm-->>Eagle: verify_logits[]
        
        Eagle->>Eagle: 树注意力验证
        Eagle->>Eagle: 计算接受的 token 数量
        
        opt 所有草稿被接受
            Eagle->>Eagle: accepted = tree_size
        end
        
        opt 部分草稿被接受
            Eagle->>Eagle: accepted = k (k < tree_size)
            Eagle->>Sampler: sample(verify_logits[k])
            Sampler-->>Eagle: corrected_token
        end
        
        opt 草稿全部被拒绝
            Eagle->>Eagle: accepted = 0
            Eagle->>Sampler: sample(verify_logits[0])
            Sampler-->>Eagle: next_token
        end
        
        Eagle->>Eagle: 更新 KVCache
        Eagle->>Eagle: 输出接受的 tokens
    end
    
    Eagle-->>Llm: generated_tokens[]
```

## 4. 后端算子执行流程

### 4.1 CPU 后端卷积执行

```mermaid
sequenceDiagram
    participant Pipeline
    participant CPUBackend
    participant CPUConvolution
    participant Tensor

    Pipeline->>CPUBackend: onCreate(inputs, outputs, op)
    CPUBackend->>CPUConvolution: new CPUConvolution(backend, op)
    CPUConvolution->>CPUConvolution: 解析卷积参数
    CPUConvolution-->>CPUBackend: Execution*
    CPUBackend-->>Pipeline: Execution*

    Pipeline->>CPUConvolution: onResize(inputs, outputs)
    CPUConvolution->>CPUConvolution: 计算输出形状
    CPUConvolution->>CPUConvolution: 选择最优实现<br/>(Winograd/Im2Col/Direct)
    CPUConvolution-->>Pipeline: ErrorCode

    Pipeline->>CPUConvolution: onExecute(inputs, outputs)
    CPUConvolution->>Tensor: input->host<float>()
    Tensor-->>CPUConvolution: float*
    CPUConvolution->>Tensor: output->host<float>()
    Tensor-->>CPUConvolution: float*
    
    CPUConvolution->>CPUConvolution: 执行卷积计算<br/>(SIMD 优化)
    CPUConvolution-->>Pipeline: ErrorCode
```

### 4.2 GPU 后端卷积执行（Metal）

```mermaid
sequenceDiagram
    participant Pipeline
    participant MetalBackend
    participant MetalConvolution
    participant MTLDevice
    participant MTLCommandQueue

    Pipeline->>MetalBackend: onCreate(inputs, outputs, op)
    MetalBackend->>MetalConvolution: new MetalConvolution(backend, op)
    MetalConvolution->>MTLDevice: newBufferWithLength(weight_size)
    MTLDevice-->>MetalConvolution: MTLBuffer*
    MetalConvolution->>MetalConvolution: 上传权重到 GPU
    MetalConvolution-->>MetalBackend: Execution*
    MetalBackend-->>Pipeline: Execution*

    Pipeline->>MetalConvolution: onResize(inputs, outputs)
    MetalConvolution->>MetalConvolution: 计算输出形状
    MetalConvolution->>MetalConvolution: 选择 Metal Shader
    MetalConvolution-->>Pipeline: ErrorCode

    Pipeline->>MetalBackend: onExecuteBegin()
    MetalBackend->>MTLCommandQueue: commandBuffer()
    MTLCommandQueue-->>MetalBackend: MTLCommandBuffer*

    Pipeline->>MetalConvolution: onExecute(inputs, outputs)
    MetalConvolution->>MetalConvolution: 编码 GPU 命令
    MetalConvolution->>MTLCommandQueue: commit()
    MetalConvolution-->>Pipeline: ErrorCode

    Pipeline->>MetalBackend: onExecuteEnd()
    MetalBackend->>MTLCommandQueue: waitUntilCompleted()
    MetalBackend-->>Pipeline: 完成
```

## 5. 内存管理流程

### 5.1 动态内存分配和复用

```mermaid
sequenceDiagram
    participant Pipeline
    participant Backend
    participant BufferAllocator
    participant MemObj

    Note over Pipeline: === 第一次分配 ===
    Pipeline->>Backend: onAcquire(tensor1, DYNAMIC)
    Backend->>BufferAllocator: alloc(size=1024, separate=false)
    BufferAllocator->>BufferAllocator: 检查空闲列表
    BufferAllocator->>BufferAllocator: 分配新内存
    BufferAllocator-->>Backend: MemChunk{ptr, size}
    Backend->>MemObj: new MemObj(chunk, allocator)
    MemObj-->>Backend: MemObj*
    Backend-->>Pipeline: MemObj*

    Note over Pipeline: === 释放到池 ===
    Pipeline->>Backend: onReleaseBuffer(tensor1, DYNAMIC)
    Backend->>MemObj: release()
    MemObj->>BufferAllocator: free(chunk)
    BufferAllocator->>BufferAllocator: 加入空闲列表
    BufferAllocator-->>MemObj: 完成
    MemObj-->>Backend: 完成

    Note over Pipeline: === 复用内存 ===
    Pipeline->>Backend: onAcquire(tensor2, DYNAMIC)
    Backend->>BufferAllocator: alloc(size=1024, separate=false)
    BufferAllocator->>BufferAllocator: 从空闲列表查找
    BufferAllocator->>BufferAllocator: 复用已有内存
    BufferAllocator-->>Backend: MemChunk{ptr, size}
    Backend->>MemObj: new MemObj(chunk, allocator)
    MemObj-->>Backend: MemObj*
    Backend-->>Pipeline: MemObj*

    Note over Pipeline: === 清理所有内存 ===
    Pipeline->>Backend: onClearBuffer()
    Backend->>BufferAllocator: release(allRelease=true)
    BufferAllocator->>BufferAllocator: 释放所有内存
    BufferAllocator-->>Backend: 完成
    Backend-->>Pipeline: 完成
```

## 时序图说明

### 1. Session API 推理流程
展示了低级 API 的完整流程：
- **模型加载**：从文件加载 FlatBuffers 模型
- **会话创建**：自动选择后端，构建执行计划
- **Resize**：形状推导和内存分配
- **执行**：算子逐个执行

### 2. Module API 推理流程
展示了高级 API 的使用：
- **模型加载**：通过 Module 接口加载
- **前向传播**：使用 VARP 进行计算

### 3. LLM 推理流程
展示了 LLM 特有的流程：
- **初始化**：加载模型、分词器、采样器
- **Prefill + Decode**：两阶段生成
- **推测解码**：EAGLE 加速策略

### 4. 后端算子执行
展示了不同后端的执行差异：
- **CPU**：同步执行，直接计算
- **GPU**：异步执行，命令队列

### 5. 内存管理
展示了动态内存的分配和复用机制，提高内存利用率。
