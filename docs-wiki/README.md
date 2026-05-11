# MNN 技术文档索引

本目录包含 MNN 项目的详细技术文档，涵盖架构设计、开发指南和最佳实践。

## 📚 文档列表

### 架构和设计

1. **[MNN_architecture.md](./MNN_architecture.md)** - MNN 仓库实现分析
   - 整体架构概览
   - 核心推理流水线
   - 后端架构详解
   - LLM 子系统
   - Diffusion 支持
   - 模型转换器
   - 量化支持

2. **[MNN-Class-Diagrams.md](./MNN-Class-Diagrams.md)** - 核心类图
   - 核心推理类图（Interpreter、Session、Pipeline、Backend）
   - Express API 类图（VARP、Module、Executor）
   - LLM 子系统类图（Llm、Tokenizer、Sampler）
   - 后端实现类图（CPU、Metal、CUDA、OpenCL、Vulkan）
   - 内存管理类图
   - 算子注册类图

3. **[MNN-Sequence-Diagrams.md](./MNN-Sequence-Diagrams.md)** - 推理流程时序图
   - Session API 推理流程（加载、创建、Resize、执行）
   - Module API 推理流程（加载、前向传播）
   - LLM 推理流程（初始化、Prefill+Decode、推测解码）
   - 后端算子执行流程（CPU、GPU）
   - 内存管理流程（分配、复用、释放）

### 开发指南

4. **[MNN-Operator-Development-Guide.md](./MNN-Operator-Development-Guide.md)** - 算子开发指南
   - 算子开发概述
   - 完整开发流程
   - Schema 定义
   - Shape Inference 实现
   - Geometry Computer 实现
   - Backend Execution 实现
   - 测试和验证
   - 最佳实践

5. **[MNN-Backend-Development-Guide.md](./MNN-Backend-Development-Guide.md)** - 后端开发指南
   - 后端架构概述（RuntimeCreator → Runtime → Backend → Execution）
   - 完整开发流程
   - Runtime 实现
   - Backend 实现
   - Execution 实现
   - Dummy 后端示例
   - 性能优化
   - 测试和验证

6. **[MNN-Performance-Optimization-Guide.md](./MNN-Performance-Optimization-Guide.md)** - 性能优化指南
   - 图优化（算子融合、常量折叠）
   - 内存优化（NC4HW4、内存复用）
   - 计算优化（SIMD、多线程、Winograd）
   - 量化优化（PTQ、QAT、混合精度）
   - LLM 优化（KVCache、推测解码、Flash Attention）
   - 性能分析工具
   - 平台特定优化

### 快速参考

7. **[MNN-Quick-Start-Guide.md](./MNN-Quick-Start-Guide.md)** - 快速入门指南
   - 环境准备和编译
   - 第一个推理示例
   - 模型转换
   - 常用 API
   - Python 接口
   - LLM 快速开始

### 其他文档

8. **[MNN_core_runtime.md](./MNN_core_runtime.md)** - 核心运行时分析
9. **[MNN_paper_code_mapping.md](./MNN_paper_code_mapping.md)** - 论文与代码映射
10. **[MNN.md](./MNN.md)** - MNN 基础介绍

## 🎯 快速导航

### 我想了解...

#### 如何快速开始使用 MNN
→ 阅读 [MNN-Quick-Start-Guide.md](./MNN-Quick-Start-Guide.md)

#### MNN 的整体架构
→ 阅读 [MNN_architecture.md](./MNN_architecture.md) 第一、二章

#### 核心类的关系
→ 查看 [MNN-Class-Diagrams.md](./MNN-Class-Diagrams.md) 的类图

#### 推理的执行流程
→ 查看 [MNN-Sequence-Diagrams.md](./MNN-Sequence-Diagrams.md) 的时序图

#### 如何添加新算子
→ 阅读 [MNN-Operator-Development-Guide.md](./MNN-Operator-Development-Guide.md)

#### LLM 推理的实现
→ 阅读 [MNN_architecture.md](./MNN_architecture.md) 第五章
→ 查看 [MNN-Class-Diagrams.md](./MNN-Class-Diagrams.md) 的 LLM 类图
→ 查看 [MNN-Sequence-Diagrams.md](./MNN-Sequence-Diagrams.md) 的 LLM 时序图

#### 如何优化性能
→ 阅读 [MNN-Performance-Optimization-Guide.md](./MNN-Performance-Optimization-Guide.md)

#### 后端的实现方式
→ 阅读 [MNN_architecture.md](./MNN_architecture.md) 第三章
→ 查看 [MNN-Class-Diagrams.md](./MNN-Class-Diagrams.md) 的后端类图

## 📖 推荐阅读顺序

### 新手入门
1. [MNN-Quick-Start-Guide.md](./MNN-Quick-Start-Guide.md) - 快速入门指南（从这里开始！）
2. [MNN.md](./MNN.md) - 了解 MNN 基础
3. [MNN_architecture.md](./MNN_architecture.md) - 理解整体架构
4. [MNN-Class-Diagrams.md](./MNN-Class-Diagrams.md) - 掌握核心类关系
5. [MNN-Sequence-Diagrams.md](./MNN-Sequence-Diagrams.md) - 理解执行流程

### 算子开发者
1. [MNN_architecture.md](./MNN_architecture.md) 第九章 - Schema/Op 定义
2. [MNN-Operator-Development-Guide.md](./MNN-Operator-Development-Guide.md) - 完整开发流程
3. [../skills/add-new-op/SKILL.md](../skills/add-new-op/SKILL.md) - AI Agent 执行指南

### 后端开发者
1. [MNN_architecture.md](./MNN_architecture.md) 第三章 - 后端架构
2. [MNN-Backend-Development-Guide.md](./MNN-Backend-Development-Guide.md) - 完整开发流程
3. [MNN-Class-Diagrams.md](./MNN-Class-Diagrams.md) - 后端实现类图
4. [MNN-Sequence-Diagrams.md](./MNN-Sequence-Diagrams.md) - 后端执行流程
5. [MNN-Performance-Optimization-Guide.md](./MNN-Performance-Optimization-Guide.md) - 性能优化

### LLM 应用开发者
1. [MNN_architecture.md](./MNN_architecture.md) 第五章 - LLM 子系统
2. [MNN-Class-Diagrams.md](./MNN-Class-Diagrams.md) - LLM 类图
3. [MNN-Sequence-Diagrams.md](./MNN-Sequence-Diagrams.md) - LLM 推理流程

## 🔧 相关资源

### 代码目录
- **核心代码**: `../source/core/` - Interpreter、Session、Pipeline
- **后端实现**: `../source/backend/` - CPU、Metal、CUDA、OpenCL、Vulkan
- **Express API**: `../express/` - 高级动态图 API
- **LLM 引擎**: `../transformers/llm/engine/` - LLM 推理引擎
- **LLM 导出**: `../transformers/llm/export/` - Python 导出工具
- **Schema 定义**: `../schema/default/` - FlatBuffers 算子定义
- **测试用例**: `../test/` - 单元测试和性能测试

### AI Agent Skills
- **添加新算子**: `../skills/add-new-op/SKILL.md`
- **ARM CPU 优化**: `../skills/arm-cpu-optimize/SKILL.md`
- **支持新 LLM**: `../skills/support-new-llm/SKILL.md`
- **回顾总结**: `../skills/retrospective/SKILL.md`

### 外部资源
- [MNN 官方文档](https://www.yuque.com/mnn/cn)
- [MNN GitHub](https://github.com/alibaba/MNN)
- [FlatBuffers 文档](https://google.github.io/flatbuffers/)

## 📝 文档贡献

### 文档规范
- 使用 Markdown 格式
- 代码示例使用语法高亮
- 图表使用 Mermaid 格式
- 保持文档更新与代码同步

### 更新文档
当代码发生重大变更时，请同步更新相关文档：
- 新增算子 → 更新算子开发指南
- 架构变更 → 更新架构文档和类图
- 流程变更 → 更新时序图

## ⚠️ 受限访问

以下目录包含内部专有代码，**禁止访问**：
- `../schema/private/`
- `../source/internal/`

## 📅 文档版本

- **创建日期**: 2026-05-08
- **最后更新**: 2026-05-08
- **MNN 版本**: 基于 master 分支（commit cdeb5b07）

## 🤝 反馈和建议

如果您发现文档中的错误或有改进建议，请：
1. 提交 Issue 到 GitHub
2. 或直接提交 Pull Request

---

**注意**: 本文档集合由 AI 辅助生成，旨在帮助开发者快速理解 MNN 的架构和开发流程。如有疑问，请参考源代码和官方文档。
