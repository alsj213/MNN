# MNN 快速入门指南

本文档帮助你快速上手 MNN，从安装到运行第一个推理示例。

## 目录
1. [环境准备](#环境准备)
2. [编译 MNN](#编译-mnn)
3. [第一个推理示例](#第一个推理示例)
4. [模型转换](#模型转换)
5. [常用 API](#常用-api)
6. [下一步](#下一步)

## 环境准备

### 系统要求

| 平台 | 最低要求 |
|------|---------|
| **Linux** | Ubuntu 16.04+, GCC 4.9+ |
| **macOS** | macOS 10.13+, Xcode 9+ |
| **Windows** | Windows 10+, Visual Studio 2017+ |
| **Android** | NDK r21+, API Level 21+ |
| **iOS** | iOS 9.0+, Xcode 11+ |

### 依赖安装

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libprotobuf-dev \
    protobuf-compiler
```

#### macOS

```bash
brew install cmake protobuf
```

#### Windows

使用 Visual Studio 2017 或更高版本，并安装 CMake。

## 编译 MNN

### 1. 克隆仓库

```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
```

### 2. 基础编译

```bash
mkdir build && cd build
cmake .. \
    -DMNN_BUILD_CONVERTER=ON \
    -DMNN_BUILD_QUANTOOLS=ON
make -j$(nproc)
```

### 3. 编译选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `MNN_BUILD_CONVERTER` | 编译模型转换工具 | OFF |
| `MNN_BUILD_QUANTOOLS` | 编译量化工具 | OFF |
| `MNN_BUILD_TEST` | 编译测试用例 | OFF |
| `MNN_BUILD_LLM` | 编译 LLM 支持 | OFF |
| `MNN_METAL` | 启用 Metal 后端 | OFF |
| `MNN_OPENCL` | 启用 OpenCL 后端 | OFF |
| `MNN_CUDA` | 启用 CUDA 后端 | OFF |

### 4. 验证编译

```bash
# 运行测试
./run_test.out

# 查看转换工具
./MNNConvert --help
```

## 第一个推理示例

### 使用 Session API（低级 API）

创建 `inference_demo.cpp`：

```cpp
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>

int main() {
    // 1. 加载模型
    auto interpreter = MNN::Interpreter::createFromFile("model.mnn");
    
    // 2. 创建 Session
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 4;
    auto session = interpreter->createSession(config);
    
    // 3. 获取输入张量并填充数据
    auto input = interpreter->getSessionInput(session, nullptr);
    auto inputPtr = input->host<float>();
    for (int i = 0; i < input->elementSize(); ++i) {
        inputPtr[i] = i * 0.1f;
    }
    
    // 4. 运行推理
    interpreter->runSession(session);
    
    // 5. 获取输出
    auto output = interpreter->getSessionOutput(session, nullptr);
    auto outputPtr = output->host<float>();
    
    std::cout << "Output: ";
    for (int i = 0; i < std::min(10, output->elementSize()); ++i) {
        std::cout << outputPtr[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### 使用 Module API（高级 API，推荐）

```cpp
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>

using namespace MNN::Express;

int main() {
    // 1. 加载模型
    auto module = Module::load({"input"}, {"output"}, "model.mnn");
    
    // 2. 创建输入
    auto input = _Input({1, 3, 224, 224}, NCHW, halide_type_of<float>());
    auto inputPtr = input->writeMap<float>();
    for (int i = 0; i < input->elementSize(); ++i) {
        inputPtr[i] = i * 0.001f;
    }
    
    // 3. 运行推理
    auto output = module->onForward({input})[0];
    
    // 4. 读取结果
    auto outputPtr = output->readMap<float>();
    std::cout << "Output: ";
    for (int i = 0; i < std::min(10, output->elementSize()); ++i) {
        std::cout << outputPtr[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### 编译和运行

```bash
g++ inference_demo.cpp -o inference_demo \
    -I../include -L. -lMNN -std=c++11
./inference_demo
```

## 模型转换

### 从 ONNX 转换

```bash
./MNNConvert \
    -f ONNX \
    --modelFile model.onnx \
    --MNNModel model.mnn \
    --bizCode biz
```

### 从 PyTorch 转换

```python
# 先导出为 ONNX
import torch
model = YourModel()
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "model.onnx")
```

```bash
# 再转换为 MNN
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn
```

### 转换选项

| 选项 | 说明 |
|------|------|
| `-f` | 源格式（ONNX/TF/CAFFE/TFLITE） |
| `--modelFile` | 源模型文件 |
| `--MNNModel` | 输出 MNN 模型 |
| `--bizCode` | 业务代码标识 |
| `--fp16` | 使用 FP16 精度 |
| `--weightQuantBits` | 权重量化位数（2/4/8） |

## 常用 API

### Session API

```cpp
// 创建 Interpreter
auto interpreter = Interpreter::createFromFile("model.mnn");

// 配置 Session
ScheduleConfig config;
config.type = MNN_FORWARD_CPU;
config.numThread = 4;
auto session = interpreter->createSession(config);

// 获取输入输出
auto input = interpreter->getSessionInput(session, "input_name");
auto output = interpreter->getSessionOutput(session, "output_name");

// 调整输入尺寸
interpreter->resizeTensor(input, {1, 3, 224, 224});
interpreter->resizeSession(session);

// 运行推理
interpreter->runSession(session);
```

### Module API

```cpp
// 加载模型
auto module = Module::load({"input"}, {"output"}, "model.mnn");

// 创建输入
auto input = _Input({1, 3, 224, 224}, NCHW);

// 前向传播
auto output = module->onForward({input})[0];
```

## Python 接口

### 安装

```bash
cd pymnn
pip install .
```

### 示例

```python
import MNN
import numpy as np

# 创建 Interpreter
interpreter = MNN.Interpreter("model.mnn")
session = interpreter.createSession()

# 获取输入输出
input_tensor = interpreter.getSessionInput(session)
output_tensor = interpreter.getSessionOutput(session)

# 准备输入数据
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float, 
                       input_data, MNN.Tensor_DimensionType_Caffe)

# 拷贝数据并运行
input_tensor.copyFrom(tmp_input)
interpreter.runSession(session)

# 获取输出
tmp_output = MNN.Tensor((1, 1000), MNN.Halide_Type_Float,
                        np.zeros((1, 1000)).astype(np.float32),
                        MNN.Tensor_DimensionType_Caffe)
output_tensor.copyToHostTensor(tmp_output)
print("Output:", tmp_output.getData())
```

## LLM 快速开始

### 导出模型

```bash
cd transformers/llm/export
python llmexport.py \
    --path Qwen/Qwen2-1.5B-Instruct \
    --export mnn \
    --quant_bit 4 \
    --dst_path ./qwen2-mnn
```

### 运行推理

```bash
cd build
cmake .. -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON
make -j$(nproc)
./llm_demo ../transformers/llm/export/qwen2-mnn/config.json
```

## 性能优化提示

### 选择后端

```cpp
config.type = MNN_FORWARD_CPU;    // CPU
config.type = MNN_FORWARD_METAL;  // iOS/macOS GPU
config.type = MNN_FORWARD_OPENCL; // Android/Linux GPU
config.type = MNN_FORWARD_CUDA;   // NVIDIA GPU
```

### 使用量化

```bash
./MNNConvert -f ONNX --modelFile model.onnx --MNNModel model.mnn \
    --weightQuantBits 8
```

## 下一步

### 深入学习
- 📖 [架构文档](./MNN_architecture.md)
- 🎨 [类图](./MNN-Class-Diagrams.md)
- 📊 [时序图](./MNN-Sequence-Diagrams.md)

### 开发指南
- 🔧 [算子开发](./MNN-Operator-Development-Guide.md)
- 🖥️ [后端开发](./MNN-Backend-Development-Guide.md)
- ⚡ [性能优化](./MNN-Performance-Optimization-Guide.md)

### 获取帮助
- [GitHub Issues](https://github.com/alibaba/MNN/issues)
- [官方文档](https://www.yuque.com/mnn/cn)
- [示例代码](../../demo/)
