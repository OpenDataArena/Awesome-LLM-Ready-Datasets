# Awesome-LLM-Ready-Datasets
[🌍 English README](./README.md)
> 为开源大模型训练准备的高价值数据集清单，覆盖文本、代码、图像/音频/视频、多模态与 Agent。  
> 采用：一级（模型/模态）→ 二级（训练阶段）→ 三级（标签）组织方式。

<p align="center">
  <img src="./llm-ready-datasets.png" alt="llm-datasets" width="700">
</p>


## 📋 目录
- [分类体系说明](#分类体系说明)
- **数据集目录**
  - [文本 Text](#文本-text)
    - [预训练 Pretraining](#text-pretraining)
    - [指令微调 Instruction Tuning](#text-instruction-tuning)
    - [对齐/RLHF Alignment/RLHF](#text-alignment-rlhf)
    - [评测/基准 Evaluation/Benchmark](#text-evaluation-benchmark)
    - [检索/RAG Retrieval/RAG](#text-retrieval-rag)
  - [代码 Code](#代码-code)
    - [预训练 Pretraining](#code-pretraining)
    - [指令微调 Instruction Tuning](#code-instruction-tuning)
    - [对齐/RLHF Alignment/RLHF](#code-alignment-rlhf)
    - [评测/基准 Evaluation/Benchmark](#code-evaluation-benchmark)
    - [检索/RAG Retrieval/RAG](#code-retrieval-rag)
  - [多模态 Multimodal](#多模态-multimodal)
    - [预训练](#multimodal-pretraining)
    - [指令微调](#multimodal-instruction-tuning)
    - [对齐/RLHF](#multimodal-alignment-rlhf)
    - [评测/基准](#multimodal-evaluation-benchmark)
    - [检索/RAG](#multimodal-retrieval-rag)
  - [生成 Generation（图像/视频/音频）](#生成-gen)
    - [预训练](#gen-pretraining)
    - [指令微调](#gen-instruction-tuning)
    - [对齐/RLHF](#gen-alignment-rlhf)
    - [评测/基准](#gen-evaluation-benchmark)
    - [检索/RAG](#gen-retrieval-rag)
  - [代理 Agent](#代理-agent)
    - [预训练](#agent-pretraining)
    - [指令微调](#agent-instruction-tuning)
    - [对齐/RLHF](#agent-alignment-rlhf)
    - [评测/基准](#agent-evaluation-benchmark)
    - [检索/RAG](#agent-retrieval-rag)
- [贡献指南](#贡献指南)
- 当前版本：v0.1

---

## 分类体系说明
**一级（模型/模态）**：Text / Code / Multimodal / Gen（图像/视频/音频生成）/ Agent  
**二级（训练阶段）**：Pretraining / Instruction Tuning / Alignment(RLHF) / Evaluation / Retrieval(RAG)  
**三级（标签）**：任务/模态细节/语言等均用标签表示，可多选：
GeneralLM, Dialogue, InstructionFollowing, MathReasoning, CodeGeneration, CodeRepair,
ImageEditing, VisionLanguageAlignment, RetrievalAugmentedGeneration,
VideoGeneration, VideoEditing, AudioGeneration, AudioUnderstanding, AudioVisualGeneration,
TextOnly, Image-Text, Audio-Text, Video-Text, CodeOnly,
English, Chinese, Multilingual, LowResource


**归类原则**
- 先按**一级→二级**分组，再给出**标签**。
- 同一数据集可在多个一级下重复列出，并在说明中注明“亦适用于…”。

---

## 数据集目录

<a id="文本-text"></a>
### 文本 Text

<a id="text-pretraining"></a>
#### 预训练 Pretraining
- **[数据集-X](link)** — 标签：`GeneralLM`, `TextOnly`, `English` — 通用大规模文本语料…
- **[数据集-Y](link)** — 标签：`MathReasoning`, `TextOnly`, `Chinese` — 中文数学题库…

<a id="text-instruction-tuning"></a>
#### 指令微调 Instruction Tuning
- **[数据集-Z](link)** — 标签：`InstructionFollowing`, `Dialogue`, `TextOnly`, `Multilingual` — 多语言指令-响应…

<a id="text-alignment-rlhf"></a>
#### 对齐/RLHF Alignment/RLHF
- *(待补充)*

<a id="text-evaluation-benchmark"></a>
#### 评测/基准 Evaluation/Benchmark
- *(待补充)*

<a id="text-retrieval-rag"></a>
#### 检索/RAG Retrieval/RAG
- *(待补充)*

---

<a id="代码-code"></a>
### 代码 Code

<a id="code-pretraining"></a>
#### 预训练 Pretraining
- **[数据集-A](link)** — 标签：`CodeGeneration`, `CodeOnly`, `Multilingual` — 大规模开源代码库…

<a id="code-instruction-tuning"></a>
#### 指令微调 Instruction Tuning
- **[数据集-B](link)** — 标签：`CodeRepair`, `InstructionFollowing`, `CodeOnly`, `English` — 代码修复指令-响应…

<a id="code-alignment-rlhf"></a>
#### 对齐/RLHF
- *(待补充)*

<a id="code-evaluation-benchmark"></a>
#### 评测/基准
- *(待补充)*

<a id="code-retrieval-rag"></a>
#### 检索/RAG
- *(待补充)*

---

<a id="多模态-multimodal"></a>
### 多模态 Multimodal

<a id="multimodal-pretraining"></a>
#### 预训练
- **[数据集-C](link)** — 标签：`VisionLanguageAlignment`, `Image-Text`, `English` — 图像-文本对齐大规模语料…

<a id="multimodal-instruction-tuning"></a>
#### 指令微调
- **[数据集-D](link)** — 标签：`ImageEditing`, `Image-Text`, `English` — 指令驱动图像编辑…

<a id="multimodal-alignment-rlhf"></a>
#### 对齐/RLHF
- *(待补充)*

<a id="multimodal-evaluation-benchmark"></a>
#### 评测/基准
- *(待补充)*

<a id="multimodal-retrieval-rag"></a>
#### 检索/RAG
- *(待补充)*

---

<a id="生成-gen"></a>
### 生成 Generation（图像/视频/音频）

<a id="gen-pretraining"></a>
#### 预训练
- **[数据集-E](link)** — 标签：`GeneralLM`, `Image-Text`, `English` — 用于图像生成预训练…

<a id="gen-instruction-tuning"></a>
#### 指令微调
- **[数据集-F](link)** — 标签：`ImageEditing`, `Image-Text`, `English` — 指令→图像编辑对…

<a id="gen-alignment-rlhf"></a>
#### 对齐/RLHF
- *(待补充)*

<a id="gen-evaluation-benchmark"></a>
#### 评测/基准
- *(待补充)*

<a id="gen-retrieval-rag"></a>
#### 检索/RAG
- *(待补充)*

---

<a id="代理-agent"></a>
### 代理 Agent

<a id="agent-pretraining"></a>
#### 预训练
- **[数据集-G](link)** — 标签：`InstructionFollowing`, `ToolUse`, `English` — 工具调用/轨迹数据…

<a id="agent-instruction-tuning"></a>
#### 指令微调
- **[数据集-H](link)** — 标签：`Dialogue`, `English`, `Multilingual` — 对话代理微调语料…

<a id="agent-alignment-rlhf"></a>
#### 对齐/RLHF
- *(待补充)*

<a id="agent-evaluation-benchmark"></a>
#### 评测/基准
- *(待补充)*

<a id="agent-retrieval-rag"></a>
#### 检索/RAG
- *(待补充)*

---

## 贡献指南
请在提交条目时包含：
- 一级（模型/模态）、二级（训练阶段）
- 三级标签（任务/模态细节/语言…）
- 链接 + 一句话简介
- 若适用于多个一级类型，请备注“亦适用于…”
