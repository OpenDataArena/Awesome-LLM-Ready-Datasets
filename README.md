# Awesome-LLM-Ready-Datasets
[🇨🇳 中文版 README](./README_zh.md)
> A curated collection of datasets ready for training large language or multimodal models — including text, code, image/video/audio domains.

<p align="center">
  <img src="./llm-ready-datasets.png" alt="llm-datasets" width="700">
</p>

## Quick Navigation  
- [Classification Framework](#classification-framework)  
- **Dataset Index**  
  - [Text](#text)  
    - [Pretraining](#text-pretraining)  
    - [Instruction Tuning / SFT](#text-instruction-tuning)  
    - [Alignment / RL](#text-alignment--rlhf)  
    - [Evaluation / Benchmark](#text-evaluation--benchmark)  
    - [Retrieval / RAG](#text-retrieval--rag)  
  - [Code](#code)  
    - [Pretraining](#code-pretraining)  
    - [Instruction Tuning / SFT](#code-instruction-tuning)  
    - [Alignment / RL](#code-alignment--rlhf)  
    - [Evaluation / Benchmark](#code-evaluation--benchmark)  
    - [Retrieval / RAG](#code-retrieval--rag)  
  - [Multimodal](#multimodal)  
    - [Pretraining](#multimodal-pretraining)  
    - [Instruction Tuning / SFT](#multimodal-instruction-tuning)  
    - [Alignment / RL](#multimodal-alignment--rlhf)  
    - [Evaluation / Benchmark](#multimodal-evaluation--benchmark)  
    - [Retrieval / RAG](#multimodal-retrieval--rag)  
  - [Generation (Image/Video/Audio)](#gen-imagevideoaudio-generation)  
    - [Pretraining](#gen-imagevideoaudio-generation-pretraining)  
    - [Instruction Tuning / SFT](#gen-imagevideoaudio-generation-instruction-tuning)  
    - [Alignment / RL](#gen-imagevideoaudio-generation-alignment--rlhf)  
    - [Evaluation / Benchmark](#gen-imagevideoaudio-generation-evaluation--benchmark)  
    - [Retrieval / RAG](#gen-imagevideoaudio-generation-retrieval--rag)  
  - [Agent](#agent)  
    - [Pretraining](#agent-pretraining)  
    - [Instruction Tuning / SFT](#agent-instruction-tuning)  
    - [Alignment / RL](#agent-alignment--rlhf)  
    - [Evaluation / Benchmark](#agent-evaluation--benchmark)  
    - [Retrieval / RAG](#agent-retrieval--rag)  
- [Contribution Guide](#contribution-guide)  
- Current Version: v0.1

---

## Classification Framework  
### Level 1: Model / Modality Type  
- **Text** — Models primarily handling text input/output.  
- **Code** — Models specialized on source code tasks (generation, repair, understanding).  
- **Multimodal** — Models handling two or more modalities (e.g., image+text, audio+text, video+text).  
- **Gen (Image/Video/Audio Generation)** — Models focused on generative tasks for image, video, or audio.  
- **Agent** — Models with interactive, tool-using, decision-making capabilities.

### Level 2: Training Stage  
- **Pretraining** — Foundational model training phase.  
- **Instruction Tuning** — Fine-tuning with instruction-response format.  
- **Alignment / RLHF** — Behavior alignment or preference learning phase.  
- **Evaluation / Benchmark** — Datasets designed for model evaluation/benchmarking.  
- **Retrieval / RAG** — Datasets for retrieval-augmented generation or knowledge integration.

### Level 3: Tags  
Tags convey additional details: task, modality details, language, usage. A dataset may have multiple tags. Examples include:  GeneralLM, Dialogue, InstructionFollowing, MathReasoning, CodeGeneration, CodeRepair, ImageEditing, VisionLanguageAlignment, RetrievalAugmentedGeneration, VideoGeneration, VideoEditing, AudioGeneration, AudioUnderstanding, AudioVisualGeneration, TextOnly, Image-Text, Audio-Text, Video-Text, CodeOnly, English, Chinese, Multilingual

- If a dataset is generic pretraining material, tag with `GeneralLM`.  
- If task is specific (e.g., video generation), tag with `VideoGeneration`.  
- If dataset involves audio+visual joint data, tag `AudioVisualGeneration`.  
- If language is a key feature, tag `English`, `Chinese`, `Multilingual`, etc.

### Classification Principles  
- Entries are first grouped by Level 1 → Level 2.  
- Each dataset entry includes Level 3 tags.  
- If a dataset is suitable for multiple Level 1 types, it can be listed under each relevant section with identical info and a note “also suitable for …”.

---

## Tag Examples Per Model / Modality Type  
### Text  
Tags: `GeneralLM`, `Dialogue`, `InstructionFollowing`, `MathReasoning`, `QA`, `TextOnly`, `English`, `Chinese`, `Multilingual`

### Code  
Tags: `CodeGeneration`, `CodeRepair`, `InstructionFollowing`, `CodeUnderstanding`, `CodeOnly`, `Multilingual`

### Multimodal  
Tags: `VisionLanguageAlignment`, `ImageEditing`, `AudioUnderstanding`, `AudioVisualGeneration`, `Image-Text`, `Audio-Text`, `Video-Text`, `Multilingual`

### Generation (Image/Video/Audio)  
Tags: `ImageEditing`, `VideoGeneration`, `VideoEditing`, `AudioGeneration`, `AudioVisualGeneration`, `Image-Text`, `Video-Text`, `Audio-Text`

### Agent  
Tags: `Dialogue`, `ToolUse`, `InstructionFollowing`, `RetrievalAugmentedGeneration`, `DecisionMaking`, `English`, `Multilingual`

---

## Dataset Index

<a id="text"></a>
### Text

<a id="text-pretraining"></a>
#### Pretraining
- **[Dataset-X](link)** — Tags: `GeneralLM`, `TextOnly`, `English` — A large-scale general text corpus…
- **[Dataset-Y](link)** — Tags: `MathReasoning`, `TextOnly`, `Chinese` — Chinese mathematics problem dataset…
- **[UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)** - Tasks: `CodeGeneration`, `InstructionFollowing` | Mod: `Image-Text`, `Video-Text` | Lang: `English`, `Chinese` |  [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceH4%2Fultrachat_200k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) [![GitHub Stars](https://img.shields.io/github/stars/WangRongsheng/awesome-LLM-resources?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/WangRongsheng/awesome-LLM-resources)
  
<a id="text-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[Dataset-Z](link)** — Tags: `InstructionFollowing`, `Dialogue`, `TextOnly`, `Multilingual` — Multilingual instruction–response pairs…

<a id="text-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*

<a id="text-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- *(add entries)*

<a id="text-retrieval-rag"></a>
#### Retrieval / RAG
- *(add entries)*

---

<a id="code"></a>
### Code

<a id="code-pretraining"></a>
#### Pretraining
- **[Dataset-A](link)** — Tags: `CodeGeneration`, `CodeOnly`, `Multilingual` — Large open-source code corpus…

<a id="code-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[Dataset-B](link)** — Tags: `CodeRepair`, `InstructionFollowing`, `CodeOnly`, `English` — Code repair instruction–response…

<a id="code-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*

<a id="code-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- *(add entries)*

<a id="code-retrieval-rag"></a>
#### Retrieval / RAG
- *(add entries)*

---

<a id="multimodal"></a>
### Multimodal

<a id="multimodal-pretraining"></a>
#### Pretraining
- **[Dataset-C](link)** — Tags: `VisionLanguageAlignment`, `Image-Text`, `English` — Large image–text alignment corpus…

<a id="multimodal-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[Dataset-D](link)** — Tags: `ImageEditing`, `Image-Text`, `English` — Instruction-driven image editing…

<a id="multimodal-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*

<a id="multimodal-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- *(add entries)*

<a id="multimodal-retrieval-rag"></a>
#### Retrieval / RAG
- *(add entries)*

---

<a id="gen"></a>
### Generation (Image/Video/Audio)

<a id="gen-pretraining"></a>
#### Pretraining
- **[Dataset-E](link)** — Tags: `GeneralLM`, `Image-Text`, `English` — Pretraining for image generation…

<a id="gen-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[Dataset-F](link)** — Tags: `ImageEditing`, `Image-Text`, `English` — Instruction→image editing pairs…

<a id="gen-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*

<a id="gen-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- *(add entries)*

<a id="gen-retrieval-rag"></a>
#### Retrieval / RAG
- *(add entries)*

---

<a id="agent"></a>
### Agent

<a id="agent-pretraining"></a>
#### Pretraining
- **[Dataset-G](link)** — Tags: `InstructionFollowing`, `ToolUse`, `English` — Tool-use traces for agent pretraining…

<a id="agent-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[Dataset-H](link)** — Tags: `Dialogue`, `English`, `Multilingual` — Dialogue data for agents…

<a id="agent-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*

<a id="agent-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- *(add entries)*

<a id="agent-retrieval-rag"></a>
#### Retrieval / RAG
- *(add entries)*
---

## Contribution Guide  
We welcome contributions of new datasets or improvements to existing entries. Please ensure your submission includes:  
- Level 1 type (Model/Modality)  
- Level 2 stage (Training Stage)  
- Level 3 tags (task, modality-detail, language, etc)  
- Dataset link & brief description  
- If dataset suits multiple Level 1 types, note “also suitable for …”

---

Thank you to all contributors. We hope this index becomes a key resource for large and multimodal model development.  


