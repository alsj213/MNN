# MNN - Deep Learning Inference Framework

## Overview

MNN is a lightweight, high-performance deep learning inference engine developed by Alibaba Group. It provides a complete runtime for loading, optimizing, and executing neural network models across diverse backends (CPU, GPU, NPU). The framework is designed around a modular architecture that separates model representation, scheduling, memory management, and backend-specific execution.

### Key Capabilities
- **Multi-Backend Support**: CPU (ARM/x86 with SIMD), OpenCL, Vulkan, Metal, CUDA, MUSA, NN-API
- **Model Formats**: MNN's native FlatBuffers-based format, with converters from TensorFlow, Caffe, ONNX, TFLite
- **Memory Optimization**: Deferred and eager allocation strategies with buffer reuse
- **Execution Scheduling**: Pipeline-based scheduling with geometry optimization
- **Control Flow**: Support for If, While, NMS, and Mixture-of-Experts operations
- **Plugin System**: Extensible shape inference and compute kernel registration
- **Quantization**: INT8, FP16, INT4 sparse weight support

---

## Architecture Overview

```mermaid
graph TB
    subgraph "API Layer"
        Interpreter[Interpreter]
        Module[Module / NetModule]
        Expr[Expr / Variable / VARP]
        ImageProcess[ImageProcess]
    end

    subgraph "Session & Scheduling"
        Session[Session]
        Schedule[Schedule]
        Pipeline[Pipeline]
    end

    subgraph "Execution & Backend"
        Execution[Execution]
        Backend[Backend]
        Runtime[Runtime]
        WrapExecution[WrapExecution]
    end

    subgraph "Memory Management"
        BufferAllocator[BufferAllocator]
        EagerAllocator[EagerBufferAllocator]
        DeferAllocator[DeferBufferAllocator]
        AutoStorage[AutoStorage / BufferStorage]
    end

    subgraph "Core Data Structures"
        Tensor[Tensor]
        Command[Command / CommandBuffer]
        Op[Op / OpT]
    end

    subgraph "Plugin System"
        PluginContext[PluginContext]
        ComputeKernel[ComputeKernel]
        InferShapeKernel[InferShapeKernel]
    end

    subgraph "Utilities"
        FileLoader[FileLoader]
        WorkerThread[WorkerThread]
        AutoTime[Timer / AutoTime]
        TensorUtils[TensorUtils]
        OpCommonUtils[OpCommonUtils]
        ConvolutionCommon[ConvolutionCommon]
    end

    Interpreter --> Session
    Interpreter --> Schedule
    Module --> PipelineModule
    Module --> StaticModule
    Expr --> Executor
    Session --> Pipeline
    Pipeline --> Execution
    Pipeline --> Backend
    Schedule --> Pipeline
    Runtime --> Backend
    Backend --> Execution
    Backend --> BufferAllocator
    Execution --> Tensor
    Command --> Execution
    WrapExecution --> Execution
```

---

## Module Structure

The MNN framework is organized into the following sub-modules:

| Sub-Module | Description | Documentation |
|---|---|---|
| **Core Runtime** | Backend, Runtime, Execution, and scheduling infrastructure | [Core Runtime](MNN_core_runtime.md) |
| **Expression System** | High-level graph IR, lazy evaluation executor, and module abstraction | [Expression System](MNN_expression_system.md) |
| **Control Flow Modules** | IfModule, WhileModule, NMSModule, MoEModule | [Control Flow Modules](MNN_control_flow.md) |
| **Memory Management** | BufferAllocator, eager/deferred strategies, AutoStorage | [Memory Management](MNN_memory_management.md) |
| **Plugin System** | Plugin context, compute kernel and shape inference registration | [Plugin System](MNN_plugin_system.md) |
| **Utilities** | File I/O, threading, timing, tensor utilities, convolution helpers | [Utilities](MNN_utilities.md) |

---

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Interpreter
    participant Schedule
    participant Session
    participant Pipeline
    participant Backend
    participant Execution

    User->>Interpreter: createFromFile / createFromBuffer
    Interpreter->>Interpreter: Parse FlatBuffers model
    User->>Interpreter: createSession(config)
    Interpreter->>Schedule: schedule(net, configs, runtime)
    Schedule->>Pipeline: Create pipeline info
    Interpreter->>Session: new Session(info, modes, runtime)
    User->>Interpreter: resizeSession(session)
    Session->>Pipeline: encode() + allocMemory()
    Pipeline->>Backend: onAcquireBuffer for tensors
    Pipeline->>Execution: Create per-op executions
    User->>Interpreter: runSession(session)
    Session->>Pipeline: execute()
    Pipeline->>Execution: onExecute(inputs, outputs)
    Execution-->>Pipeline: Results
    Pipeline-->>Session: Done
    Session-->>User: Output tensors ready
```

### Alternative: Expression API Flow

```mermaid
sequenceDiagram
    participant User
    participant Variable
    participant Expr
    participant Executor
    participant Session

    User->>Variable: load / create (build graph)
    Variable->>Expr: Create expression nodes
    User->>Executor: compute / prepareCompute
    Executor->>Executor: makeCache (schedule graph)
    Executor->>Session: Create internal Session
    Session->>Session: resize + run
    Session-->>User: Results via Variable::readMap
```

---

## Key Design Patterns

### 1. Backend Abstraction
All hardware-specific code lives behind the `Backend` and `Runtime` interfaces. `Runtime` factories create `Backend` instances, which in turn create `Execution` objects for individual ops.

### 2. Schedule-Pipeline-Execution Chain
Models flow through: **Schedule** → **Pipeline** → **Execution**:
- `Schedule` partitions the model graph into pipeline stages
- `Pipeline` encodes ops, allocates memory, and orchestrates execution
- `Execution` performs the actual computation

### 3. Lazy Evaluation (Expression System)
The `Expr`/`Variable`/`VARP` system builds a computation graph lazily. `Executor` triggers scheduling and execution only when results are requested via `readMap()`.

### 4. Memory Reuse
`BufferAllocator` with both eager and deferred strategies reuses memory across tensors. Deferred mode defers actual allocation until the total size is known, enabling single-allocation.

---

## Core Data Types

| Type | File | Purpose |
|---|---|---|
| `halide_buffer_t` | HalideRuntime.h | Raw buffer descriptor (host + device pointers, dims, type) |
| `halide_type_t` | HalideRuntime.h | Type descriptor (code, bits, lanes) |
| `Tensor` | Tensor.hpp | Data container wrapping `halide_buffer_t` |
| `Op` / `OpT` | FlatBuffers generated | Operation descriptor |
| `Command` | Command.hpp | Runnable unit: op + tensors + execution |
| `MNNForwardType` | MNNForwardType.h | Backend type enum |
| `BackendConfig` | MNNForwardType.h | Backend configuration (memory, power, precision modes) |

---

## Cross-References

- **[Core Runtime](MNN_core_runtime.md)**: `Backend`, `Runtime`, `Execution`, `Session`, `Schedule`, `Pipeline`, `Tensor`, `Command`
- **[Expression System](MNN_expression_system.md)**: `Expr`, `Variable`, `VARP`, `Executor`, `Module`, `PipelineModule`, `StaticModule`
- **[Control Flow Modules](MNN_control_flow.md)**: `IfModule`, `WhileModule`, `NMSModule`, `MoEModule`
- **[Memory Management](MNN_memory_management.md)**: `BufferAllocator`, `EagerBufferAllocator`, `DeferBufferAllocator`, `AutoStorage`
- **[Plugin System](MNN_plugin_system.md)**: `PluginContext`, `ComputeKernel`, `InferShapeKernel`
- **[Utilities](MNN_utilities.md)**: `FileLoader`, `WorkerThread`, `AutoTime`, `TensorUtils`, `OpCommonUtils`, `ConvolutionCommon`
