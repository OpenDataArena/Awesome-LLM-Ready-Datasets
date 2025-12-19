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
- **[ArabicText 2022](https://data.baai.ac.cn/datadetail/ArabicText-2022)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Arabic`

- **[BNC](https://www.natcorp.ox.ac.uk/)** - Tasks: `Text Classification`, `Information Extraction` and so on | Mod: `Text` | Lang: `English`
  
- **[Baidu baike](https://huggingface.co/datasets/xuqinyang/BaiduBaike-5.63M)** - Tasks: `Information Extraction`, `Question Answering` | Mod: `Text` | Lang: `Chinese` |  [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fxuqinyang%2FBaiduBaike-5.63M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/xuqinyang/BaiduBaike-5.63M) [![GitHub Stars](https://img.shields.io/github/stars/BIT-ENGD/baidu_baike?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/BIT-ENGD/baidu_baike)
  
- **CLUECorpus2020** - Mod: `Text` | Lang: `Multi`

- **DuSQL** - Tasks: `Code Generation` | Mod: `Text`, `Code` | Lang: `Chinese`

- **Gutenberg project** - Tasks: `Text Classification`, `Summarization` | Mod: `Text` | Lang: `Multi`

- **Huggingface dataset** - Mod: `Text` | Lang: `English`

- **LCSTS** - Tasks: `Summarization` | Mod: `Text` | Lang: `Chinese`

- **LongForm** - Tasks: `Instruction-Following`, `Summarization` | Mod: `Text` | Lang: `English`

- **MultiUN** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi`

- **News-crawl** - Tasks: `Text Classification`, `Information Extraction`, `Summarization` | Mod: `Text` | Lang: `English`, `Multi`

- **OpenWebText** - Tasks: `Text Classification`, `Summarization`, `Dialogue` | Mod: `Text` | Lang: `English`

- **ParaCrawl** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi`

- **Project Gutenberg** - Tasks: `Text Classification`, `Summarization` | Mod: `Text` | Lang: `English`

- **PubMed Central** - Tasks: `Information Extraction`, `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `English`

- **PushshPairs reddit** - Tasks: `Dialogue`, `Information Extraction` | Mod: `Text` | Lang: `English`

- **Pushshift Reddit** - Tasks: `Dialogue`, `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `English`, `Multi`

- **Reddit** - Tasks: `Dialogue`, `Information Extraction`, `Text Classification` | Mod: `Text` | Lang: `English`

- **Smashwords** - Tasks: `Text Classification`, `Summarization`, `Information Extraction` | Mod: `Text` | Lang: `English`

- **StackExchange** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English`

- **TensorFlow dataset** - Mod: `Text` | Lang: `English`

- **Toronto Book Corpus** - Mod: `Text` | Lang: `English`

- **UNCorpus v1.0** - Mod: `Text` | Lang: `Multi`

- **WuDaoCorpora-Text** - Mod: `Text` | Lang: `Chinese`

- **Zhihu** - Tasks: `Dialogue`, `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `Chinese`

- **[Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPT4all)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FQingyiSi%2FAlpaca-CoT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/GPT4all) [![GitHub Stars](https://img.shields.io/github/stars/nomic-ai/gpt4all?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nomic-ai/gpt4all)

- **[AmericanStories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)** - Tasks: `Text Classification`, `Information Extraction`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdell-research-harvard%2FAmericanStories&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)

- **[CBook-150K](https://github.com/FudanNLPLAB/CBook-150K)** - Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/FudanNLPLAB/CBook-150K?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FudanNLPLAB/CBook-150K)

- **[CLUECorpus](https://github.com/CLUEbenchmark/CLUE)** - Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUE)

- **[CSLDCP](https://github.com/CLUEbenchmark/FewCLUE)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/FewCLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/FewCLUE)

- **[Cabrita Dataset](https://huggingface.co/datasets/cabrita-labs/cabrita-instruct-52k)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Portuguese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcabrita-labs%2Fcabrita-instruct-52k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cabrita-labs/cabrita-instruct-52k)

- **[Chatgpt_corpus](https://github.com/PlexPt/chatgpt-corpus/releases/tag/3)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/PlexPt/chatgpt-corpus?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/PlexPt/chatgpt-corpus/releases/tag/3)

- **[ChineseWebText2.0](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0)** - Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FCASIA-LM%2FChineseWebText2.0&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0) [![GitHub Stars](https://img.shields.io/github/stars/CASIA-LM/ChineseWebText-2.0?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CASIA-LM/ChineseWebText-2.0)

- **[ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)** - Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FCASIA-LM%2FChineseWebText&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/CASIA-LM/ChineseWebText) [![GitHub Stars](https://img.shields.io/github/stars/CASIA-LM/ChineseWebText?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CASIA-LM/ChineseWebText)

- **[Common Crawl](https://github.com/facebookresearch/cc_net)** - Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/cc_net?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/cc_net)

- **[Expository-Prose-V1](https://huggingface.co/datasets/pints-ai/Expository-Prose-V1)** - Tasks: `Text Classification`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fpints-ai%2FExpository-Prose-V1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/pints-ai/Expository-Prose-V1) [![GitHub Stars](https://img.shields.io/github/stars/Pints-AI/1.5-Pints?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Pints-AI/1.5-Pints)

- **[FinNLP](https://github.com/AI4Finance-Foundation/FinNLP)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinNLP?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AI4Finance-Foundation/FinNLP)

- **[Finance](https://huggingface.co/datasets/gbharti/finance-alpaca)** - Tasks: `Instruction-Following`, `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fgbharti%2Ffinance-alpaca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/gbharti/finance-alpaca)

- **[Future-Idea-Generation](https://github.com/sandeep82945/Future-Idea-Generation)** - Tasks: `Instruction-Following`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/sandeep82945/Future-Idea-Generation?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sandeep82945/Future-Idea-Generation)

- **[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44)** - Tasks: `Machine Translation`, `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FCohereLabs%2Finclude-base-44&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/CohereLabs/include-base-44)

- **[InstructionTranslation](https://huggingface.co/datasets/Instruction-Tuning-with-GPT-4/Instruction-Translation)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FInstruction-Tuning-with-GPT-4%2FInstruction-Translation&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Instruction-Tuning-with-GPT-4/Instruction-Translation) [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/m2m-12b?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/m2m-12b)

- **[LaMini-instruction](https://huggingface.co/datasets/MBZUAI/LaMini-instruction)** - Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMBZUAI%2FLaMini-instruction&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MBZUAI/LaMini-instruction) [![GitHub Stars](https://img.shields.io/github/stars/mbzuai-nlp/LaMini-LM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mbzuai-nlp/LaMini-LM)

- **[LawGPT_zh](https://github.com/LiuHC0428/LAW-GPT)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/LiuHC0428/LAW-GPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LiuHC0428/LAW-GPT)

- **[Long Form](https://github.com/akoksal/LongForm)** - Tasks: `Text Generation` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/akoksal/LongForm?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/akoksal/LongForm)

- **[LongWriter-6k](https://huggingface.co/datasets/THUDM/LongWriter-6k)** - Tasks: `Instruction-Following`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTHUDM%2FLongWriter-6k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/THUDM/LongWriter-6k) [![GitHub Stars](https://img.shields.io/github/stars/THUDM/LongWriter?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/LongWriter)

- **[MCS-Bench](https://github.com/SCUT-DLVCLab/MCS-Bench)** - Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/SCUT-DLVCLab/MCS-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SCUT-DLVCLab/MCS-Bench)

- **[MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTIGER-Lab%2FMathInstruct&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) [![GitHub Stars](https://img.shields.io/github/stars/TIGER-AI-Lab/MAmmoTH?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TIGER-AI-Lab/MAmmoTH)

- **[MathPile](https://huggingface.co/datasets/GAIR/MathPile)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGAIR%2FMathPile&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/GAIR/MathPile) [![GitHub Stars](https://img.shields.io/github/stars/GAIR-NLP/MathPile?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/GAIR-NLP/MathPile)

- **[Medical Meadow](https://github.com/kbressem/medAlpaca)** - Tasks: `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/kbressem/medAlpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/kbressem/medAlpaca)

- **[NLLB](https://github.com/facebookresearch/fairseq/tree/nllb)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/fairseq?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/fairseq/tree/nllb)

- **[NMBVC](https://github.com/esbatmop/MNBVC)** - Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/esbatmop/MNBVC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/esbatmop/MNBVC)

- **[NaturalProofs](https://github.com/wellecks/naturalproofs)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/wellecks/naturalproofs?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/wellecks/naturalproofs)

- **[OSCAR-2201](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201)** - Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Foscar-corpus%2FOSCAR-2201&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201)

- **[OpenOrca (full)](https://huggingface.co/datasets/Open-Orca/OpenOrca)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpen-Orca%2FOpenOrca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Open-Orca/OpenOrca) [![GitHub Stars](https://img.shields.io/github/stars/OpenOrca/OpenOrca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenOrca/OpenOrca)

- **[OpenR1‑Math‑220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopen-r1%2FOpenR1-Math-220k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)

- **[PRM800K](https://github.com/openai/prm800k)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/openai/prm800k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/prm800k)

- **[RealNews](https://github.com/rowanz/grover)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/rowanz/grover?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/rowanz/grover)

- **[RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)** - Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftogethercomputer%2FRedPajama-Data-1T&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) [![GitHub Stars](https://img.shields.io/github/stars/ryanlewis322/RedPajama?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ryanlewis322/RedPajama)

- **[RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)** - Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftogethercomputer%2FRedPajama-Data-V2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) [![GitHub Stars](https://img.shields.io/github/stars/togethercomputer/RedPajama-Data?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/togethercomputer/RedPajama-Data)

- **[S2ORC](https://github.com/allenai/s2orc)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/allenai/s2orc?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/s2orc)

- **[ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `English`, `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FshareAI%2FShareGPT-Chinese-English-90k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) [![GitHub Stars](https://img.shields.io/github/stars/CrazyBoyM/llama2-Chinese-chat?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CrazyBoyM/llama2-Chinese-chat)

- **[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B)** - Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcerebras%2FSlimPajama-627B&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cerebras/SlimPajama-627B)

- **[StackOverflow post](https://huggingface.co/datasets/mikex86/stackoverflow-posts)** - Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmikex86%2Fstackoverflow-posts&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mikex86/stackoverflow-posts) [![GitHub Stars](https://img.shields.io/github/stars/StackExchange/StackExchange.DataExplorer?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/StackExchange/StackExchange.DataExplorer)

- **[The Pile](https://github.com/EleutherAI/the-pile)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering`, `Reasoning`, `Summarization` | Mod: `Text` | Lang: `English`, `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/EleutherAI/the-pile?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/EleutherAI/the-pile)

- **[TigerBot Series](https://github.com/TigerResearch/TigerBot#%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%E9%9B%86)** - Tasks: `Text Classification`, `Information Extraction`, `Machine Translation` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot#%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%E9%9B%86)

- **[UltraFeedback (cleaned binarized)](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fargilla%2Fultrafeedback-binarized-preferences-cleaned&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)

- **[Unnatural Instructions](https://huggingface.co/datasets/allenai/natural-instructions)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fnatural-instructions&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/natural-instructions) [![GitHub Stars](https://img.shields.io/github/stars/EleutherAI/unnatural-instructions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/EleutherAI/unnatural-instructions)

- **[WUNT2017](https://huggingface.co/datasets/wnut_17)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwnut_17&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/wnut_17)

- **[WebText](https://github.com/openai/gpt-2)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/openai/gpt-2?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/gpt-2)

- **[WikiMatrix](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/LASER?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix)

- **[WildChat‑4.8M (nontoxic subset)](https://huggingface.co/datasets/allenai/WildChat)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2FWildChat&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/WildChat)

- **[Wizard‑LM Chinese Evol](https://huggingface.co/datasets/WizardLM/WizardLM_Chinese_instruct_dataset)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FWizardLM%2FWizardLM_Chinese_instruct_dataset&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/WizardLM/WizardLM_Chinese_instruct_dataset)

- **[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcsebuetnlp%2Fxlsum&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/csebuetnlp/xlsum)

- **[awesome chinese legal resources](https://github.com/pengxiao-song/awesome-chinese-legal-resources)** - Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/pengxiao-song/awesome-chinese-legal-resources?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/pengxiao-song/awesome-chinese-legal-resources)

- **[bookcorpusopen](https://huggingface.co/datasets/bookcorpusopen)** - Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbookcorpusopen&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bookcorpusopen) [![GitHub Stars](https://img.shields.io/github/stars/jackbandy/bookcorpus-datasheet?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jackbandy/bookcorpus-datasheet)

- **[c4](https://huggingface.co/datasets/allenai/c4)** - Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fc4&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/c4)

- **[cc100](https://huggingface.co/datasets/cc100)** - Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcc100&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cc100)

- **[dolma](https://huggingface.co/datasets/allenai/dolma)** - Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fdolma&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/dolma)

- **[falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)** - Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftiiuae%2Ffalcon-refinedweb&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)

- **[finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)** - Tasks: `Text Classification`, `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fgbharti%2Ffinance-alpaca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/gbharti/finance-alpaca)

- **[im-feeling-curious](https://huggingface.co/datasets/xiyuez/im-feeling-curious)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fxiyuez%2Fim-feeling-curious&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/xiyuez/im-feeling-curious) [![GitHub Stars](https://img.shields.io/github/stars/jonathanjohanness/im-feeling-curious?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jonathanjohanness/im-feeling-curious)

- **[mc4](https://huggingface.co/datasets/mc4)** - Tasks: `Text Classification`, `Information Extraction`, `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmc4&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mc4)

- **[nlp_Chinese_Corpus](https://github.com/brightmart/nlp_chinese_corpus)** - Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/brightmart/nlp_chinese_corpus?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/brightmart/nlp_chinese_corpus)

- **[open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopen-web-math%2Fopen-web-math&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/open-web-math/open-web-math) [![GitHub Stars](https://img.shields.io/github/stars/keirp/OpenWebMath?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/keirp/OpenWebMath)

- **[peS2o](https://huggingface.co/datasets/allenai/peS2o)** - Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2FpeS2o&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/peS2o)

- **[pretrain_en](https://huggingface.co/datasets/TigerResearch/pretrain_en)** - Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Fpretrain_en&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/pretrain_en) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)

- **[pretrain_zh](https://huggingface.co/datasets/TigerResearch/pretrain_zh)** - Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Fpretrain_zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/pretrain_zh) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)

- **[proof-pile](https://huggingface.co/datasets/hoskinson-center/proof-pile)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhoskinson-center%2Fproof-pile&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hoskinson-center/proof-pile)

- **[school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Fschool_math_0.25M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/10M)

- **[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)** - Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Ftrain_0.5M_CN&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)

- **[train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)** - Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Ftrain_1M_CN&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/train_1M_CN) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)

- **[wikipedia](https://huggingface.co/datasets/wikipedia)** - Tasks: `Question Answering`, `Summarization`, `Text Classification` | Mod: `Text` | Lang: `English`, `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwikipedia&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/wikipedia)

- **[xP3](https://huggingface.co/datasets/bigscience/xP3)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigscience%2FxP3&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigscience/xP3)

- **arXiv** - Tasks: `Text Classification`, `Information Extraction`, `Summarization` | Mod: `Text` | Lang: `English`, `Multi`

- **mOSCAR** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi`

- **paper** - Mod: `Text` | Lang: `English`



<a id="text-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **CoQA** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English`

- **DuReader 2.0** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **DuReader Yes/No** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **GSM‑IC** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **IFLYTEK** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese`

- **InstructGPT-sft** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English`

- **Opinion Abstracts** - Tasks: `Summarization` | Mod: `Text` | Lang: `English`

- **Quoref** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English`

- **ReCoRD** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **WMT** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi`

- **[0.5M version](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Ftrain_0.5M_CN&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)

- **[AESLC](https://huggingface.co/datasets/aeslc)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Faeslc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/aeslc) [![GitHub Stars](https://img.shields.io/github/stars/ryanzhumich/AESLC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ryanzhumich/AESLC)

- **[AGNEWS](https://huggingface.co/datasets/ag_news)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fag_news&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ag_news)

- **[ALLaVA-4V Data](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFreedomIntelligence%2FALLaVA-4V&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) [![GitHub Stars](https://img.shields.io/github/stars/FreedomIntelligence/ALLaVA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FreedomIntelligence/ALLaVA)

- **[AQUA-RAT](https://huggingface.co/datasets/aqua_rat)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Faqua_rat&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/aqua_rat) [![GitHub Stars](https://img.shields.io/github/stars/google-deepmind/AQuA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-deepmind/AQuA)

- **[ASDiv](https://huggingface.co/datasets/EleutherAI/asdiv)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FEleutherAI%2Fasdiv&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/EleutherAI/asdiv) [![GitHub Stars](https://img.shields.io/github/stars/chaochun/nlu-asdiv-dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/chaochun/nlu-asdiv-dataset)

- **[Adversarial QA](https://huggingface.co/datasets/adversarial_qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fadversarial_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/adversarial_qa) [![GitHub Stars](https://img.shields.io/github/stars/maxbartolo/adversarialQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/maxbartolo/adversarialQA)

- **[Alpaca Data Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fyahma%2Falpaca-cleaned&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/yahma/alpaca-cleaned) [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca)

- **[Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftatsu-lab%2Falpaca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tatsu-lab/alpaca) [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca)

- **[Alpaca GPT‑4 Chinese](https://huggingface.co/datasets/Instruction-Tuning-with-GPT-4/GPT-4-LLM)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FInstruction-Tuning-with-GPT-4%2FGPT-4-LLM&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

- **[Alpaca GPT‑4 Data](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fvicgalle%2Falpaca-gpt4&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca)

- **[Alpaca data](https://github.com/tatsu-lab/stanford_alpaca#data-release)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca#data-release)

- **[Alpaca-GPT-4_zh-cn](https://huggingface.co/datasets/shibing624/alpaca-zh)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fshibing624%2Falpaca-zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shibing624/alpaca-zh)

- **[AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/gururise/AlpacaDataCleaned?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/gururise/AlpacaDataCleaned)

- **[Alpaca_GPT4_data](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#data-release)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#data-release)

- **[Alpaca‑CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)** - Tasks: `Instruction-Following`, `Reasoning` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FQingyiSi%2FAlpaca-CoT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca)

- **[Ape210K](https://github.com/Chenny0808/ape210k)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Chenny0808/ape210k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Chenny0808/ape210k)

- **[BELLE](https://github.com/LianjiaTech/BELLE)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE)

- **[BQ](https://huggingface.co/datasets/shibing624/nli_zh)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fshibing624%2Fnli_zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shibing624/nli_zh)

- **[Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering`, `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMBZUAI%2FBactrian-X&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MBZUAI/Bactrian-X)

- **[Baize Dataset](https://github.com/project-baize/baize-chatbot/tree/main/data)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/project-baize/baize-chatbot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/project-baize/baize-chatbot/tree/main/data)

- **[C3](https://github.com/nlpdata/c3)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/nlpdata/c3?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nlpdata/c3)

- **[CLOTH](https://huggingface.co/datasets/AndyChiang/cloth)** - Tasks: `Instruction-Following`, `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FAndyChiang%2Fcloth&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/AndyChiang/cloth)

- **[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUENER2020?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUENER2020)

- **[CLiB](https://github.com/jeinlee1991/chinese-llm-benchmark)** - Tasks: `Classification`, `Information Extraction`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/jeinlee1991/chinese-llm-benchmark?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jeinlee1991/chinese-llm-benchmark)

- **[CMD](https://github.com/Toyhom/Chinese-medical-dialogue-data)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Toyhom/Chinese-medical-dialogue-data?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Toyhom/Chinese-medical-dialogue-data)

- **[CMRC2018](https://github.com/ymcui/cmrc2018)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/ymcui/cmrc2018?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ymcui/cmrc2018)

- **[CMtMedQA](https://huggingface.co/datasets/Suprit/CMtMedQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSuprit%2FCMtMedQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Suprit/CMtMedQA) [![GitHub Stars](https://img.shields.io/github/stars/SupritYoung/Zhongjing?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SupritYoung/Zhongjing)

- **[CNN-DM](https://huggingface.co/datasets/cnn_dailymail)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcnn_dailymail&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cnn_dailymail)

- **[CNewSum](https://github.com/dqwang122/MLROUGE)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/dqwang122/MLROUGE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/dqwang122/MLROUGE)

- **[CUAD](https://huggingface.co/datasets/cuad)** - Tasks: `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcuad&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cuad)

- **[ChatAlpaca data](https://github.com/cascip/ChatAlpaca)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/cascip/ChatAlpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/cascip/ChatAlpaca)

- **[ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Kent0n-Li/ChatDoctor?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Kent0n-Li/ChatDoctor)

- **[ChatMed_Consult_Dataset](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmichaelwzhu%2FChatMed_Consult_Dataset&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset) [![GitHub Stars](https://img.shields.io/github/stars/michael-wzhu/ChatMed?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/michael-wzhu/ChatMed)

- **[Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Fchatbot_arena_conversations&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)

- **[Child_chat_data](https://github.com/HIT-SCIR-SC/QiaoBan)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/HIT-SCIR-SC/QiaoBan?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HIT-SCIR-SC/QiaoBan)

- **[CommonGen](https://huggingface.co/datasets/common_gen)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcommon_gen&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/common_gen) [![GitHub Stars](https://img.shields.io/github/stars/INK-USC/CommonGen?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/INK-USC/CommonGen)

- **[CrossWOZ](https://github.com/thu-coai/CrossWOZ)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/thu-coai/CrossWOZ?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-coai/CrossWOZ)

- **[DART](https://huggingface.co/datasets/dart)** - Tasks: `Text Generation` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdart&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/dart) [![GitHub Stars](https://img.shields.io/github/stars/Yale-LILY/dart?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Yale-LILY/dart)

- **[DISC-Fin-SFT](https://github.com/FudanDISC/DISC-FinLLM)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/FudanDISC/DISC-FinLLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FudanDISC/DISC-FinLLM)

- **[DISC-Law-SFT](https://github.com/FudanDISC/DISC-LawLLM)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/FudanDISC/DISC-LawLLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FudanDISC/DISC-LawLLM)

- **[DISC-Med-SFT](https://huggingface.co/datasets/Flmc/DISC-Med-SFT)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFlmc%2FDISC-Med-SFT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Flmc/DISC-Med-SFT) [![GitHub Stars](https://img.shields.io/github/stars/FudanDISC/DISC-MedLLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FudanDISC/DISC-MedLLM)

- **[DREAM](https://github.com/nlpdata/dream)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nlpdata/dream?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nlpdata/dream)

- **[DialogStudio](https://github.com/salesforce/DialogStudio)** - Tasks: `Dialogue`, `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/salesforce/DialogStudio?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/salesforce/DialogStudio)

- **[Dialogue RE](https://github.com/nlpdata/dialogre)** - Tasks: `Dialogue`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nlpdata/dialogre?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nlpdata/dialogre)

- **[DocRED](https://github.com/thunlp/DocRED)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/DocRED?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/DocRED)

- **[Dolly‑15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdatabricks%2Fdatabricks-dolly-15k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/databricks/databricks-dolly-15k) [![GitHub Stars](https://img.shields.io/github/stars/databricks/dolly-15k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/databricks/dolly-15k)

- **[DuoRC](https://huggingface.co/datasets/duorc)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fduorc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/duorc)

- **[Dynosaur](https://huggingface.co/datasets/YUWEI995/dynosaur)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FYUWEI995%2Fdynosaur&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/YUWEI995/dynosaur)

- **[E2E](https://huggingface.co/datasets/e2e_nlg?row=0)** - Tasks: `Dialogue`, `Text Generation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fe2e_nlg&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/e2e_nlg?row=0) [![GitHub Stars](https://img.shields.io/github/stars/tuetschek/e2e-dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tuetschek/e2e-dataset)

- **[ELI5](https://github.com/facebookresearch/ELI5)** - Tasks: `Question Answering`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/ELI5?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/ELI5)

- **[EPRSTMT](https://github.com/CLUEbenchmark/FewCLUE)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/FewCLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/FewCLUE)

- **[Firefly(流萤)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FYeungNLP%2Ffirefly-train-1.1M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)

- **[Firefly](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)** - Tasks: `Instruction-Following`, `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FYeungNLP%2Ffirefly-train-1.1M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)

- **[GPTeacher](https://github.com/teknium1/GPTeacher)** - Tasks: `Instruction-Following`, `Dialogue`, `Roleplay` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/teknium1/GPTeacher?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/teknium1/GPTeacher)

- **[HEAD-QA](https://huggingface.co/datasets/head_qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhead_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/head_qa) [![GitHub Stars](https://img.shields.io/github/stars/aghie/head-qa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/aghie/head-qa)

- **[IGN Clean Instruct 500K](https://huggingface.co/datasets/teknium/ign_clean_instruct_dataset_500k)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fteknium%2Fign_clean_instruct_dataset_500k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/teknium/ign_clean_instruct_dataset_500k)

- **[IMDB](https://huggingface.co/datasets/imdb)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fimdb&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/imdb)

- **[IWSLT 2017](https://huggingface.co/datasets/iwslt2017)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `English`, `French`, `German`, `Spanish` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fiwslt2017&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/iwslt2017)

- **[Infinity‑Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBAAI%2FInfinity-Instruct&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BAAI/Infinity-Instruct)

- **[InstructDial](https://github.com/prakharguptaz/Instructdial)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/prakharguptaz/Instructdial?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/prakharguptaz/Instructdial)

- **[InstructDoc](https://github.com/nttmdlab-nlp/InstructDoc)** - Tasks: `Instruction-Following`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nttmdlab-nlp/InstructDoc?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nttmdlab-nlp/InstructDoc)

- **[InstructionWild](https://huggingface.co/datasets/XueFuzhao/InstructionWild)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English`, `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FXueFuzhao%2FInstructionWild&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/XueFuzhao/InstructionWild) [![GitHub Stars](https://img.shields.io/github/stars/InstructionWild/InstructionWild?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/InstructionWild/InstructionWild)

- **[Japanese Alpaca](https://huggingface.co/datasets/studioml-staging/japanese-alpaca-data)** - Tasks: `Instruction-Following`, `Dialogue`, `Text Classification`, `Machine Translation` | Mod: `Text` | Lang: `Japanese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstudioml-staging%2Fjapanese-alpaca-data&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/studioml-staging/japanese-alpaca-data) [![GitHub Stars](https://img.shields.io/github/stars/Tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Tatsu-lab/stanford_alpaca)

- **[Lawyer LLaMA_sft](https://github.com/AndrewZhe/lawyer-llama)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/AndrewZhe/lawyer-llama?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AndrewZhe/lawyer-llama)

- **[Luotuo-QA-B](https://huggingface.co/datasets/Logic123456789/Luotuo-QA-B)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`, `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FLogic123456789%2FLuotuo-QA-B&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Logic123456789/Luotuo-QA-B) [![GitHub Stars](https://img.shields.io/github/stars/LC1332/Luotuo-QA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LC1332/Luotuo-QA)

- **[MARC](https://huggingface.co/datasets/amazon_reviews_multi)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Famazon_reviews_multi&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/amazon_reviews_multi)

- **[METS-CoV](https://github.com/YLab-Open/METS-CoV)** - Tasks: `Text Classification`, `Sentiment Analysis` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/YLab-Open/METS-CoV?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/YLab-Open/METS-CoV)

- **[MOSS SFT data](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)** - Tasks: `Dialogue`, `Text Classification` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/OpenLMLab/MOSS?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data)

- **[MS MARCO](https://huggingface.co/datasets/ms_marco)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fms_marco&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ms_marco) [![GitHub Stars](https://img.shields.io/github/stars/microsoft/MSMARCO-Question-Answering?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/microsoft/MSMARCO-Question-Answering)

- **[Math23K](https://github.com/SCNU203/Math23k)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/SCNU203/Math23k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SCNU203/Math23k)

- **[MeChat data](https://github.com/qiuhuachuan/smile)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/qiuhuachuan/smile?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/qiuhuachuan/smile)

- **[MedDialog](https://github.com/UCSD-AI4H/Medical-Dialogue-System)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/UCSD-AI4H/Medical-Dialogue-System?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/UCSD-AI4H/Medical-Dialogue-System)

- **[MediaSum](https://huggingface.co/datasets/ccdv/mediasum)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fccdv%2Fmediasum&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ccdv/mediasum) [![GitHub Stars](https://img.shields.io/github/stars/zcgzcgzcg1/MediaSum?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/zcgzcgzcg1/MediaSum/)

- **[MultiNews](https://huggingface.co/datasets/multi_news)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmulti_news&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/multi_news) [![GitHub Stars](https://img.shields.io/github/stars/Alex-Fabbri/Multi-News?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Alex-Fabbri/Multi-News)

- **[MultiWOZ](https://github.com/budzianowski/multiwoz)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/budzianowski/multiwoz?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/budzianowski/multiwoz)

- **[NATURAL INSTRUCTIONS](https://github.com/allenai/natural-instructions)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/allenai/natural-instructions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/natural-instructions)

- **[Natural Questions](https://huggingface.co/datasets/natural_questions)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnatural_questions&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/natural_questions) [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/natural-questions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/natural-questions)

- **[OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenAssistant%2Foasst1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenAssistant/oasst1) [![GitHub Stars](https://img.shields.io/github/stars/LAION-AI/OASST1?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LAION-AI/OASST1)

- **[OASST2 (final)](https://huggingface.co/datasets/OpenAssistant/oasst2)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenAssistant%2Foasst2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenAssistant/oasst2) [![GitHub Stars](https://img.shields.io/github/stars/LAION-AI/Open-Assistant?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LAION-AI/Open-Assistant)

- **[OIG](https://huggingface.co/datasets/laion/OIG)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text`, `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flaion%2FOIG&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/laion/OIG)

- **[OntoNotes 5.0](https://huggingface.co/datasets/conll2012_ontonotesv5)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fconll2012_ontonotesv5&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/conll2012_ontonotesv5)

- **[OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)** - Tasks: `Instruction-Following`, `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnvidia%2FOpenMathInstruct-1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) [![GitHub Stars](https://img.shields.io/github/stars/Kipok/NeMo-Skills?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Kipok/NeMo-Skills)

- **[OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpen-Orca%2FOpenOrca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Open-Orca/OpenOrca) [![GitHub Stars](https://img.shields.io/github/stars/OpenOrca/OpenOrca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenOrca/OpenOrca)

- **[Open‑PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmlabonne%2Fopen-perfectblend&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mlabonne/open-perfectblend)

- **[PromptSource](https://github.com/bigscience-workshop/promptsource)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/bigscience-workshop/promptsource?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/bigscience-workshop/promptsource)

- **[PsyQA](https://github.com/thu-coai/PsyQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/thu-coai/PsyQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-coai/PsyQA)

- **[QuAC](https://huggingface.co/datasets/quac)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fquac&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/quac)

- **[RefGPT](https://huggingface.co/datasets/xusenlinzy/refgpt-1.0)** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fxusenlinzy%2Frefgpt-1.0&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/xusenlinzy/refgpt-1.0) [![GitHub Stars](https://img.shields.io/github/stars/RefGPT/RefGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/RefGPT/RefGPT)

- **[Resume](https://github.com/jiesutd/LatticeLSTM)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/jiesutd/LatticeLSTM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jiesutd/LatticeLSTM)

- **[SAMSum](https://huggingface.co/datasets/samsum)** - Tasks: `Summarization`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsamsum&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/samsum)

- **[SHP](https://huggingface.co/datasets/stanfordnlp/SHP)** - Tasks: `Dialogue`, `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstanfordnlp%2FSHP&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stanfordnlp/SHP)

- **[SUPER-NATURAL INSTRUCTIONS](https://github.com/allenai/natural-instructions)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/allenai/natural-instructions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/natural-instructions)

- **[Sentiment140](https://huggingface.co/datasets/sentiment140)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsentiment140&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/sentiment140)

- **[ShareGPT_ Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fanon8231489123%2FShareGPT_Vicuna_unfiltered&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)

- **[Spider](https://github.com/taoyds/spiderhttps://github.com/taoyds/spider)** - Tasks: `Code Generation`, `Question Answering` | Mod: `Text`, `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/taoyds/spiderhttps:?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/taoyds/spiderhttps://github.com/taoyds/spider)

- **[TACRED](https://huggingface.co/datasets/DFKI-SLT/tacred)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FDFKI-SLT%2Ftacred&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/DFKI-SLT/tacred)

- **[TAPIR‑Cleaned](https://huggingface.co/datasets/voidful/Tapir-Cleaned)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fvoidful%2FTapir-Cleaned&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/voidful/Tapir-Cleaned)

- **[THUCNews](https://github.com/thunlp/THUCTC)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/THUCTC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/THUCTC)

- **[TNEWS](https://github.com/CLUEbenchmark/CLUE)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUE)

- **[TSI-v0](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftasksource%2Ftasksource-instruct-v0&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0)

- **[Taobao NER](https://github.com/allanj/ner_incomplete_annotation)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/allanj/ner_incomplete_annotation?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allanj/ner_incomplete_annotation)

- **[TheoremQA](https://huggingface.co/datasets/wenhu/TheoremQA)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwenhu%2FTheoremQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/wenhu/TheoremQA) [![GitHub Stars](https://img.shields.io/github/stars/Great-Expectations/TheoremQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Great-Expectations/TheoremQA)

- **[Traditional Chinese Alpaca](https://huggingface.co/datasets/voidful/alpaca-trad-chinese)** - Tasks: `Instruction-Following`, `Dialogue`, `Machine Translation` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fvoidful%2Falpaca-trad-chinese&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/voidful/alpaca-trad-chinese)

- **[UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceH4%2Fultrachat_200k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

- **[V2](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FWizardLM%2FWizardLM_evol_instruct_V2_196k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)

- **[Vicuna Dataset](https://huggingface.co/datasets/lmsys/vicuna)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Fvicuna&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/vicuna) [![GitHub Stars](https://img.shields.io/github/stars/lmsys/vicuna?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lmsys/vicuna)

- **[WebMedQA](https://github.com/hejunqing/webMedQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/hejunqing/webMedQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hejunqing/webMedQA)

- **[Weibo NER](https://github.com/hltcoe/golden-horse)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/hltcoe/golden-horse?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hltcoe/golden-horse)

- **[WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/mahnazkoupaee/WikiHow-Dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mahnazkoupaee/WikiHow-Dataset)

- **[WildChat](https://huggingface.co/datasets/allenai/WildChat)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2FWildChat&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/WildChat)

- **[WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fwildguardmix&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/wildguardmix)

- **[WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fwildjailbreak&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/wildjailbreak)

- **[Wizard-LM-Chinese-instruct-evol](https://huggingface.co/datasets/silk-road/Wizard-LM-Chinese-instruct-evol)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsilk-road%2FWizard-LM-Chinese-instruct-evol&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/silk-road/Wizard-LM-Chinese-instruct-evol) [![GitHub Stars](https://img.shields.io/github/stars/LC1332/Chinese-alpaca-lora?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LC1332/Chinese-alpaca-lora)

- **[WizardLM Evol‑Instruct V2](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FWizardLM%2FWizardLM_evol_instruct_V2_196k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)

- **[WizardLM evol](https://huggingface.co/datasets/WizardLM/evol-instruct)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FWizardLM%2Fevol-instruct&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/WizardLM/evol-instruct) [![GitHub Stars](https://img.shields.io/github/stars/WizardLM/WizardLM-evolution?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/WizardLM/WizardLM-evolution)

- **[Youku NER](https://github.com/allanj/ner_incomplete_annotation)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/allanj/ner_incomplete_annotation?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allanj/ner_incomplete_annotation)

- **[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fshibing624%2Falpaca-zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shibing624/alpaca-zh) [![GitHub Stars](https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#data-release)

- **[alpaca_chinese dataset](https://github.com/hikariming/alpaca_chinese_dataset)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/hikariming/alpaca_chinese_dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hikariming/alpaca_chinese_dataset)

- **[arxiv‑math‑instruct‑50k (ArtifactAI)](https://huggingface.co/datasets/ArtifactAI/arxiv-math-instruct-50k)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FArtifactAI%2Farxiv-math-instruct-50k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ArtifactAI/arxiv-math-instruct-50k)

- **[arxiv‑math‑instruct‑50k](https://huggingface.co/datasets/LIAMF-USP/arxiv-math-instruct-50k)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FLIAMF-USP%2Farxiv-math-instruct-50k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/LIAMF-USP/arxiv-math-instruct-50k)

- **[blended_skill_talk](https://github.com/facebookresearch/blended_skill_talk)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/blended_skill_talk?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/blended_skill_talk)

- **[chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Fchatbot_arena_conversations&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)

- **[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdatabricks%2Fdatabricks-dolly-15k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/databricks/databricks-dolly-15k) [![GitHub Stars](https://img.shields.io/github/stars/databricks/dolly-15k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/databricks/dolly-15k)

- **[generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Fgenerated_chat_0.4M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/10M)

- **[glaive‑function‑calling‑v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)** - Tasks: `Tool-Use` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fglaiveai%2Fglaive-function-calling-v2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)

- **[im-feeling- curious](https://huggingface.co/datasets/xiyuez/im-feeling-curious)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fxiyuez%2Fim-feeling-curious&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/xiyuez/im-feeling-curious)

- **[kollm-converations](https://huggingface.co/datasets/davidkim205/kollm-converations)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdavidkim205%2Fkollm-converations&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/davidkim205/kollm-converations)

- **[lima](https://huggingface.co/datasets/GAIR/lima)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English`, `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGAIR%2Flima&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/GAIR/lima)

- **[lithuanian-qa-v1](https://huggingface.co/datasets/neurotechnology/lithuanian-qa-v1)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Lt` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fneurotechnology%2Flithuanian-qa-v1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/neurotechnology/lithuanian-qa-v1)

- **[lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Flmsys-chat-1m&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)

- **[math](https://huggingface.co/datasets/ArtifactAI/arxiv-math-instruct-50k)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FArtifactAI%2Farxiv-math-instruct-50k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ArtifactAI/arxiv-math-instruct-50k)

- **[medical](https://huggingface.co/datasets/shibing624/medical)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fshibing624%2Fmedical&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shibing624/medical) [![GitHub Stars](https://img.shields.io/github/stars/shibing624/MedicalGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/shibing624/MedicalGPT)

- **[multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Fmultiturn_chat_0.8M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/10M)

- **[no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceH4%2Fno_robots&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceH4/no_robots)

- **[orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmicrosoft%2Forca-math-word-problems-200k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)

- **[orca‑chat](https://huggingface.co/datasets/Open-Orca/OpenOrca)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpen-Orca%2FOpenOrca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Open-Orca/OpenOrca)

- **[self-instruct](https://github.com/yizhongw/self-instruct)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/yizhongw/self-instruct?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/yizhongw/self-instruct)

- **[sft_en](https://huggingface.co/datasets/TigerResearch/sft_en)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Fsft_en&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/sft_en) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)

- **[sft_zh](https://huggingface.co/datasets/TigerResearch/sft_zh)** - Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Fsft_zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/sft_zh) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)

- **[smolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceTB%2Fsmoltalk&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) [![GitHub Stars](https://img.shields.io/github/stars/smolTalk/smolTalk?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/smolTalk/smolTalk)

- **[stack-exchange-paired](https://huggingface.co/datasets/stanfordnlp/stack-exchange-paired)** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstanfordnlp%2Fstack-exchange-paired&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stanfordnlp/stack-exchange-paired)

- **[tigerbot-law-plugin](https://huggingface.co/datasets/TigerResearch/tigerbot-law-plugin)** - Tasks: `Legal`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Ftigerbot-law-plugin&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/tigerbot-law-plugin) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)

- **[webglm-qa](https://huggingface.co/datasets/THUDM/webglm-qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTHUDM%2Fwebglm-qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/THUDM/webglm-qa) [![GitHub Stars](https://img.shields.io/github/stars/THUDM/WebGLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/WebGLM)

- **[webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2Fwebgpt_comparisons&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/webgpt_comparisons)

- **cMedQA2** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **ign_clean _instruct _dataset_500k** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English`



<a id="text-alignment-rlhf"></a>
#### Alignment / RL
- **[Anthropic HH Golden](https://huggingface.co/datasets/Anthropic/hh-rlhf)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FAnthropic%2Fhh-rlhf&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Anthropic/hh-rlhf)

- **[CS](https://huggingface.co/datasets/ArtifactAI/arxiv-beir-cs-ml-generated-queries)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FArtifactAI%2Farxiv-beir-cs-ml-generated-queries&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ArtifactAI/arxiv-beir-cs-ml-generated-queries)

- **[Download](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

- **[FineGrainedRLHF](https://github.com/allenai/FineGrainedRLHF)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/allenai/FineGrainedRLHF?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/FineGrainedRLHF)

- **[Flan V2](https://github.com/google-research/FLAN/tree/main/flan/v2)** - Tasks: `Instruction-Following`, `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research/FLAN?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research/FLAN/tree/main/flan/v2)

- **[GPT-4-LLM Dataset](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)

- **[HH‑RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FAnthropic%2Fhh-rlhf&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Anthropic/hh-rlhf)

- **[HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnvidia%2FHelpSteer2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nvidia/HelpSteer2) [![GitHub Stars](https://img.shields.io/github/stars/HelpSteer/HelpSteer2?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HelpSteer/HelpSteer2)

- **[OpenAI Summarization Comparison](https://huggingface.co/datasets/openai/summarize_from_feedback)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2Fsummarize_from_feedback&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/summarize_from_feedback)

- **[OpenAI WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2Fwebgpt_comparisons&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/webgpt_comparisons)

- **[PKU‑SafeRLHF‑10K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPKU-Alignment%2FPKU-SafeRLHF-10K&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K)

- **[Physics](https://huggingface.co/datasets/ArtifactAI/arxiv-physics-instruct-tune-30k)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FArtifactAI%2Farxiv-physics-instruct-tune-30k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ArtifactAI/arxiv-physics-instruct-tune-30k)

- **[Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fgarage-bAInd%2FOpen-Platypus&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)

- **[WebGPT (comparisons)](https://huggingface.co/datasets/openai/webgpt_comparisons)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2Fwebgpt_comparisons&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/webgpt_comparisons)

- **[WizardLM evolve_instruct V2](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FWizardLM%2FWizardLM_evol_instruct_V2_196k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)

- **[chatbot_arena _conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Fchatbot_arena_conversations&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)

- **[dolphin](https://huggingface.co/datasets/ehartford/dolphin)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fehartford%2Fdolphin&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ehartford/dolphin)

- **[helpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnvidia%2FHelpSteer&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nvidia/HelpSteer)

- **[hh-rlhf](https://github.com/anthropics/hh-rlhf)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/anthropics/hh-rlhf?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/anthropics/hh-rlhf)

- **[nonofficial link](https://github.com/sufengniu/RefGPT)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/sufengniu/RefGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sufengniu/RefGPT)

- **[oasst1_pairwise_rlhf_reward](https://huggingface.co/datasets/tasksource/oasst1_pairwise_rlhf_reward)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftasksource%2Foasst1_pairwise_rlhf_reward&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tasksource/oasst1_pairwise_rlhf_reward)

- **[on Huggingface](https://huggingface.co/datasets/Anthropic/hh-rlhf)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FAnthropic%2Fhh-rlhf&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Anthropic/hh-rlhf)

- **[pku-saferlhf-dataset](https://github.com/PKU-Alignment/safe-rlhf#pku-saferlhf-dataset)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/PKU-Alignment/safe-rlhf?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/PKU-Alignment/safe-rlhf#pku-saferlhf-dataset)

- **[unnatural-instructions](https://github.com/orhonovich/unnatural-instructions)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/orhonovich/unnatural-instructions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/orhonovich/unnatural-instructions)


<a id="text-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- **AFQMC** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **AIME 1983-2025 (updated annually)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **ARC** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **C<sup>3</sup> Bench 2024-5** - Tasks: `Text Classification`, `Information Extraction`, `Machine Translation`, `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **CINLID** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese`

- **CLUEWSC2020** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `Chinese`

- **COPA** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **CRAG** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English`

- **CUGE** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `Multi`

- **CoLA** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English`

- **DROP** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **DuQM** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **DuReader Checklist** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **DuReader Robust** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **FLUE** - Tasks: `Sentiment Analysis`, `Text Classification`, `Named Entity Recognition`, `Question Answering` | Mod: `Text` | Lang: `English`

- **FinBen** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English`

- **FineMath** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese`

- **HaluEval-Wild** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **LCQMC** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **LLMEVAL-1** - Tasks: `Question Answering`, `Reading Comprehension` | Mod: `Text` | Lang: `English`

- **LLMEVAL-2** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`

- **LLMEVAL-3** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi`

- **LMExamQA** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi`

- **MRPC** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English`

- **MultiMed** - Mod: `Text` | Lang: `Multi`

- **Natural Instruction** - Tasks: `Instruction-Following`, `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Multi`

- **PersianMMLU 2024-4** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Fa`

- **QASPER** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English`

- **QuAIL** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English`

- **QuaRel** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **ReClor** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English`

- **SCALE** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi`

- **SIGHAN** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese`

- **SQuAD 2.0** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English`

- **STRATEGYQA** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English`

- **SarcasmBench 2024-8** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English`

- **SuperGLUE** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`

- **WiC** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `English`

- **[ALCE](https://huggingface.co/datasets/princeton-nlp/ALCE-data)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fprinceton-nlp%2FALCE-data&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/princeton-nlp/ALCE-data) [![GitHub Stars](https://img.shields.io/github/stars/princeton-nlp/ALCE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/princeton-nlp/ALCE)

- **[ALCUNA](https://github.com/Arvid-pku/ALCUNA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Arvid-pku/ALCUNA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Arvid-pku/ALCUNA)

- **[ANLI](https://github.com/facebookresearch/anli/)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/anli?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/anli/)

- **[ARB](https://github.com/TheDuckAI/arb)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/TheDuckAI/arb?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TheDuckAI/arb)

- **[ARES](https://github.com/stanford-futuredata/ARES)** - Tasks: `Evaluation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/stanford-futuredata/ARES?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/stanford-futuredata/ARES)

- **[AlignBench](https://github.com/THUDM/AlignBench)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/THUDM/AlignBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/AlignBench)

- **[ArabLegalEval 2024-8](https://github.com/Thiqah/ArabLegalEval)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Arabic` | [![GitHub Stars](https://img.shields.io/github/stars/Thiqah/ArabLegalEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Thiqah/ArabLegalEval)

- **[ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Arabic` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMBZUAI%2FArabicMMLU&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) [![GitHub Stars](https://img.shields.io/github/stars/mbzuai-nlp/ArabicMMLU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mbzuai-nlp/ArabicMMLU)

- **[BBF-CFLEB](https://github.com/ssymmetry/BBT-FinCUGE-Applications)** - Tasks: `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/ssymmetry/BBT-FinCUGE-Applications?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ssymmetry/BBT-FinCUGE-Applications)

- **[BBH](https://github.com/suzgunmirac/BIG-Bench-Hard)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/suzgunmirac/BIG-Bench-Hard?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/suzgunmirac/BIG-Bench-Hard)

- **[BIG-Bench](https://github.com/google/BIG-bench)** - Tasks: `Reasoning`, `Common Sense Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google/BIG-bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google/BIG-bench)

- **[BIRD](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)** - Tasks: `Question Answering`, `Code Generation` | Mod: `Text`, `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/AlibabaResearch/DAMO-ConvAI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)

- **[BOSS](https://github.com/lifan-yuan/OOD_NLP)** - Tasks: `Sentiment Analysis`, `Toxicity Detection`, `Natural Language Inference`, `Named Entity Recognition`, `Extractive Question Answering` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/lifan-yuan/OOD_NLP?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lifan-yuan/OOD_NLP)

- **[BUSTM](https://github.com/xiaobu-coai/BUSTM)** - Tasks: `Semantic Matching` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/xiaobu-coai/BUSTM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/xiaobu-coai/BUSTM)

- **[BoolQ](https://github.com/google-research-datasets/boolean-questions)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/boolean-questions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/boolean-questions)

- **[C-CLUE](https://github.com/jizijing/C-CLUE)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/jizijing/C-CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jizijing/C-CLUE)

- **[CBLUE](https://github.com/CBLUEbenchmark/CBLUE)** - Tasks: `Information Extraction`, `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CBLUEbenchmark/CBLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CBLUEbenchmark/CBLUE)

- **[CG-Eval](https://huggingface.co/datasets/Besteasy/CG-Eval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBesteasy%2FCG-Eval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Besteasy/CG-Eval) [![GitHub Stars](https://img.shields.io/github/stars/Felixgithub2017/CG-Eval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Felixgithub2017/CG-Eval)

- **[CLEVA](https://github.com/LaVi-Lab/CLEVA)** - Tasks: `Evaluation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/LaVi-Lab/CLEVA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LaVi-Lab/CLEVA)

- **[CLUE](https://github.com/CLUEbenchmark/CLUE)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUE)

- **[CLongEval](https://huggingface.co/datasets/zexuanqiu22/CLongEval)** - Tasks: `Question Answering`, `Summarization`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fzexuanqiu22%2FCLongEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/zexuanqiu22/CLongEval) [![GitHub Stars](https://img.shields.io/github/stars/zexuanqiu/CLongEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/zexuanqiu/CLongEval)

- **[CMB](https://huggingface.co/datasets/FreedomIntelligence/CMB)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFreedomIntelligence%2FCMB&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/FreedomIntelligence/CMB) [![GitHub Stars](https://img.shields.io/github/stars/FreedomIntelligence/CMB?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FreedomIntelligence/CMB)

- **[CMNLI](https://github.com/CLUEbenchmark/CLUE)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUE)

- **[CMRC2019](https://github.com/ymcui/cmrc2019)** - Tasks: `Cloze Test` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/ymcui/cmrc2019?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ymcui/cmrc2019)

- **[CREAK](https://github.com/yasumasaonoe/creak)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/yasumasaonoe/creak?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/yasumasaonoe/creak)

- **[CSCD-IME](https://github.com/nghuyong/cscd-ime)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nghuyong/cscd-ime?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nghuyong/cscd-ime)

- **[CSL](https://github.com/ydli-ai/CSL)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/ydli-ai/CSL?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ydli-ai/CSL)

- **[CSpider](https://github.com/taolusi/chisp)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/taolusi/chisp?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/taolusi/chisp)

- **[ChID](https://huggingface.co/datasets/thu-coai/chid/tree/main/original)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fthu-coai%2Fchid&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/thu-coai/chid/tree/main/original) [![GitHub Stars](https://img.shields.io/github/stars/chujiezheng/ChID-Dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/chujiezheng/ChID-Dataset)

- **[Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/FranxYao/chain-of-thought-hub?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FranxYao/chain-of-thought-hub)

- **[Chinese-SimpleQA](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenStellarTeam%2FChinese-SimpleQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA) [![GitHub Stars](https://img.shields.io/github/stars/OpenStellarTeam/ChineseSimpleQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenStellarTeam/ChineseSimpleQA)

- **[ChineseFactEval](https://github.com/GAIR-NLP/factool)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/GAIR-NLP/factool?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/GAIR-NLP/factool)

- **[Choice-75](https://github.com/JoeyHou/branching)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/JoeyHou/branching?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/JoeyHou/branching)

- **[CoNLL2003](https://huggingface.co/datasets/conll2003)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fconll2003&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/conll2003)

- **[CommitmentBank](https://github.com/mcdm/CommitmentBank)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/mcdm/CommitmentBank?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mcdm/CommitmentBank)

- **[CommonsenseQA](https://github.com/jonathanherzig/commonsenseqa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/jonathanherzig/commonsenseqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jonathanherzig/commonsenseqa)

- **[CondaQA](https://huggingface.co/datasets/lasha-nlp/CONDAQA)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flasha-nlp%2FCONDAQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lasha-nlp/CONDAQA) [![GitHub Stars](https://img.shields.io/github/stars/AbhilashaRavichander/CondaQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AbhilashaRavichander/CondaQA)

- **[CosmosQA](https://huggingface.co/datasets/cosmos_qa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcosmos_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cosmos_qa) [![GitHub Stars](https://img.shields.io/github/stars/wilburOne/cosmosqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/wilburOne/cosmosqa/)

- **[Counting-Stars](https://github.com/nick7nlp/counting-stars)** - Tasks: `Long Text Task` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/nick7nlp/counting-stars?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nick7nlp/counting-stars)

- **[CrowS-Pairs](https://github.com/nyu-mll/crows-pairs)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nyu-mll/crows-pairs?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nyu-mll/crows-pairs)

- **[DPR](https://huggingface.co/datasets/definite_pronoun_resolution)** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdefinite_pronoun_resolution&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/definite_pronoun_resolution)

- **[DebateQA 2024-8](https://github.com/pillowsofwind/DebateQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/pillowsofwind/DebateQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/pillowsofwind/DebateQA)

- **[ECQA](https://github.com/dair-iitd/ECQA-Dataset)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/dair-iitd/ECQA-Dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/dair-iitd/ECQA-Dataset)

- **[EcomGPT_eval](https://github.com/Alibaba-NLP/EcomGPT)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Alibaba-NLP/EcomGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Alibaba-NLP/EcomGPT)

- **[EmotionBench](https://github.com/CUHK-ARISE/EmotionBench)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/CUHK-ARISE/EmotionBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CUHK-ARISE/EmotionBench)

- **[FACTOR](https://github.com/AI21Labs/factor)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/AI21Labs/factor?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AI21Labs/factor)

- **[FActScore](https://github.com/shmsw25/FActScore)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/shmsw25/FActScore?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/shmsw25/FActScore)

- **[FactualityPrompt](https://github.com/nayeon7lee/FactualityPrompt)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nayeon7lee/FactualityPrompt?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nayeon7lee/FactualityPrompt)

- **[FairEval](https://github.com/i-Eval/FairEval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/i-Eval/FairEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/i-Eval/FairEval)

- **[Few-NERD](https://github.com/thunlp/Few-NERD)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/Few-NERD?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/Few-NERD)

- **[FewCLUE](https://github.com/CLUEbenchmark/FewCLUE)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/FewCLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/FewCLUE)

- **[FewRel](https://github.com/thunlp/fewrel)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/fewrel?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/fewrel)

- **[FinEval](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSUFE-AIFLM-Lab%2FFinEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval) [![GitHub Stars](https://img.shields.io/github/stars/SUFE-AIFLM-Lab/FinEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SUFE-AIFLM-Lab/FinEval)

- **[FinancelQ](https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Duxiaoman-DI/XuanYuan?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ)

- **[FlagEval](https://github.com/FlagOpen/FlagEval)** - Tasks: `Choice QA`, `Classification`, `Generation QA` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/FlagOpen/FlagEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FlagOpen/FlagEval)

- **[FreshQA](https://github.com/freshllms/freshqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/freshllms/freshqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/freshllms/freshqa)

- **[GLUE-X](https://github.com/YangLinyi/GLUE-X)** - Tasks: `Sentiment Analysis`, `Linguistic Acceptability`, `Textual Similarity`, `Natural Language Inference`, `Question Answering`, `Textual Entailment`, `Paraphrase` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/YangLinyi/GLUE-X?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/YangLinyi/GLUE-X)

- **[GLUE](https://github.com/nyu-mll/GLUE-baselines)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nyu-mll/GLUE-baselines?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nyu-mll/GLUE-baselines)

- **[GSM8K](https://github.com/openai/grade-school-math)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/openai/grade-school-math?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/grade-school-math)

- **[GeoBench](https://github.com/davendw49/k2)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/davendw49/k2?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/davendw49/k2)

- **[GitHub&Download](https://github.com/allenai/natural-instructions)** - Tasks: `Instruction-Following`, `Evaluation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/allenai/natural-instructions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/natural-instructions)

- **[HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHello-SimpleAI%2FHC3&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

- **[HELM](https://github.com/stanford-crfm/helm)** - Tasks: `Question Answering`, `Information Retrieval`, `Sentiment Analysis`, `Toxicity Detection` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/stanford-crfm/helm?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/stanford-crfm/helm)

- **[HOTPOTQA](https://huggingface.co/datasets/hotpot_qa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhotpot_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hotpot_qa)

- **[HalluDial 2024-6](https://github.com/FlagOpen/HalluDial)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/FlagOpen/HalluDial?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FlagOpen/HalluDial)

- **[HalluQA](https://github.com/xiami2019/HalluQA/)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/xiami2019/HalluQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/xiami2019/HalluQA/)

- **[HaluEval](https://github.com/RUCAIBox/HaluEval)** - Tasks: `Question Answering`, `Dialogue`, `Summarization` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/RUCAIBox/HaluEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/RUCAIBox/HaluEval)

- **[HellaSwag](https://github.com/rowanz/hellaswag)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/rowanz/hellaswag?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/rowanz/hellaswag)

- **[InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)** - Tasks: `Question Answering`, `Reasoning`, `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fxinrongzhang2022%2FInfiniteBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench) [![GitHub Stars](https://img.shields.io/github/stars/OpenBMB/InfiniteBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenBMB/InfiniteBench)

- **[JEC-QA](https://github.com/thunlp/jec-qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/jec-qa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/jec-qa)

- **[KoLA](https://github.com/THU-KEG/KoLA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/THU-KEG/KoLA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THU-KEG/KoLA)

- **[LAMBADA](https://huggingface.co/datasets/lambada)** - Tasks: `Word Prediction`, `Cloze Test` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flambada&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lambada)

- **[LAiW](https://github.com/Dai-shen/LAiW)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Dai-shen/LAiW?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Dai-shen/LAiW)

- **[LEXTREME](https://github.com/JoelNiklaus/LEXTREME)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Portuguese`, `German`, `El`, `French` | [![GitHub Stars](https://img.shields.io/github/stars/JoelNiklaus/LEXTREME?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/JoelNiklaus/LEXTREME)

- **[LEval](https://huggingface.co/datasets/L4NLP/LEval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FL4NLP%2FLEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/L4NLP/LEval) [![GitHub Stars](https://img.shields.io/github/stars/OpenLMLab/LEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenLMLab/LEval)

- **[LLMEval2](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/WideDeep)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/AlibabaResearch/DAMO-ConvAI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/WideDeep)

- **[LMentry](https://github.com/aviaefrat/lmentry)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/aviaefrat/lmentry?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/aviaefrat/lmentry)

- **[LawBench](https://github.com/open-compass/LawBench)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/open-compass/LawBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/open-compass/LawBench)

- **[LeSC 2024-5](https://github.com/jinyangwu/LeSC)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/jinyangwu/LeSC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jinyangwu/LeSC)

- **[LexGLUE](https://github.com/coastalcph/lex-glue)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/coastalcph/lex-glue?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/coastalcph/lex-glue)

- **[LogiQA](https://github.com/lgw863/LogiQA-dataset)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/lgw863/LogiQA-dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lgw863/LogiQA-dataset)

- **[LongBench](https://huggingface.co/datasets/THUDM/LongBench)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English`, `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTHUDM%2FLongBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/THUDM/LongBench) [![GitHub Stars](https://img.shields.io/github/stars/LongBench/longbench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LongBench/longbench)

- **[LongEval](https://github.com/DachengLi1/LongChat)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/DachengLi1/LongChat?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/DachengLi1/LongChat)

- **[LooGLE](https://huggingface.co/datasets/bigainlco/LooGLE)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigainlco%2FLooGLE&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigainlco/LooGLE) [![GitHub Stars](https://img.shields.io/github/stars/bigai-nlco/LooGLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/bigai-nlco/LooGLE)

- **[M3KE](https://huggingface.co/datasets/TJUNLP/M3KE)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTJUNLP%2FM3KE&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TJUNLP/M3KE) [![GitHub Stars](https://img.shields.io/github/stars/tjunlp-lab/M3KE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tjunlp-lab/M3KE)

- **[MCTS](https://github.com/blcuicall/mcts)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/blcuicall/mcts?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/blcuicall/mcts)

- **[MCTest](https://huggingface.co/datasets/sagnikrayc/mctest)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsagnikrayc%2Fmctest&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/sagnikrayc/mctest)

- **[MGSM](https://huggingface.co/datasets/juletxara/mgsm)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fjuletxara%2Fmgsm&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/juletxara/mgsm) [![GitHub Stars](https://img.shields.io/github/stars/google-research/url-nlp?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research/url-nlp)

- **[MLQA](https://huggingface.co/datasets/mlqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmlqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mlqa) [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/MLQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/MLQA)

- **[MM-NIAH](https://github.com/OpenGVLab/MM-NIAH)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/MM-NIAH?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenGVLab/MM-NIAH)

- **[MMCU](https://github.com/Felixgithub2017/MMCU)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Felixgithub2017/MMCU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Felixgithub2017/MMCU)

- **[MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld)** - Tasks: `Evaluation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fyifanzhang114%2FMME-RealWorld&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) [![GitHub Stars](https://img.shields.io/github/stars/yfzhang114/MME-RealWorld?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/yfzhang114/MME-RealWorld)

- **[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTIGER-Lab%2FMMLU-Pro&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) [![GitHub Stars](https://img.shields.io/github/stars/TIGER-AI-Lab/MMLU-Pro?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TIGER-AI-Lab/MMLU-Pro)

- **[MMLU](https://github.com/hendrycks/test)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/hendrycks/test?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hendrycks/test)

- **[MMMLU](https://huggingface.co/datasets/openai/MMMLU)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2FMMMLU&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/MMMLU)

- **[MMMU](https://huggingface.co/datasets/MMMU/MMMU)** - Mod: `Text` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMMMU%2FMMMU&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MMMU/MMMU) [![GitHub Stars](https://img.shields.io/github/stars/MMMU-Benchmark/MMMU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/MMMU-Benchmark/MMMU)

- **[MMT-Bench](https://huggingface.co/datasets/Kaining/MMT-Bench)** - Tasks: `Evaluation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FKaining%2FMMT-Bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Kaining/MMT-Bench) [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/MMT-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenGVLab/MMT-Bench)

- **[MSRA](https://huggingface.co/datasets/msra_ner)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmsra_ner&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/msra_ner)

- **[MathQA](https://huggingface.co/datasets/math_qa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmath_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/math_qa)

- **[MedNLI](https://huggingface.co/datasets/bigbio/mednli)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigbio%2Fmednli&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigbio/mednli) [![GitHub Stars](https://img.shields.io/github/stars/jgc128/mednli?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jgc128/mednli)

- **[MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmeta-math%2FMetaMathQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/meta-math/MetaMathQA) [![GitHub Stars](https://img.shields.io/github/stars/meta-math/MetaMath?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/meta-math/MetaMath)

- **[MiniF2F_v1](https://github.com/openai/miniF2F)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/openai/miniF2F?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/miniF2F)

- **[MultiNLI](https://huggingface.co/datasets/multi_nli)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmulti_nli&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/multi_nli)

- **[MultiRC](https://huggingface.co/datasets/eraser_multi_rc)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Feraser_multi_rc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/eraser_multi_rc) [![GitHub Stars](https://img.shields.io/github/stars/CogComp/multirc?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CogComp/multirc)

- **[MultiTrust](https://github.com/thu-ml/MMTrustEval)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/thu-ml/MMTrustEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-ml/MMTrustEval)

- **[NAH (Needle-in-a-Haystack)](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/gkamradt/LLMTest_NeedleInAHaystack?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

- **[NaturalReasoning](https://huggingface.co/datasets/facebook/natural_reasoning)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ffacebook%2Fnatural_reasoning&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/facebook/natural_reasoning) [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/Natural-Reasoning?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/Natural-Reasoning)

- **[NeedleBench 2024-7](https://github.com/open-compass/opencompass)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/open-compass/opencompass?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/open-compass/opencompass)

- **[NeuLR](https://github.com/deepreasoning/neulr)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/deepreasoning/neulr?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/deepreasoning/neulr)

- **[Newsroom](https://huggingface.co/datasets/newsroom)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnewsroom&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/newsroom)

- **[OCNLI](https://github.com/cluebenchmark/OCNLI)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/cluebenchmark/OCNLI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/cluebenchmark/OCNLI)

- **[OlympiadBench](https://github.com/OpenBMB/OlympiadBench)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/OpenBMB/OlympiadBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenBMB/OlympiadBench)

- **[OpenBookQA](https://huggingface.co/datasets/openbookqa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenbookqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openbookqa) [![GitHub Stars](https://img.shields.io/github/stars/allenai/OpenBookQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/OpenBookQA)

- **[Owl-Bench](https://github.com/HC-Guo/Owl)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/HC-Guo/Owl?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HC-Guo/Owl)

- **[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx)** - Tasks: `Semantic Matching` | Mod: `Text` | Lang: `English`, `Chinese`, `Spanish`, `French`, `German`, `Russian`, `Japanese`, `Korean`, `Arabic`, `Hindi`, `Portuguese`, `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/paws?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/paws/tree/master/pawsx)

- **[PAWS](https://huggingface.co/datasets/paws)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fpaws&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/paws) [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/paws?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/paws)

- **[PIQA](https://huggingface.co/datasets/piqa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fpiqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/piqa) [![GitHub Stars](https://img.shields.io/github/stars/francois-rozet/piqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/francois-rozet/piqa)

- **[PROST](https://huggingface.co/datasets/corypaik/prost)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcorypaik%2Fprost&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/corypaik/prost) [![GitHub Stars](https://img.shields.io/github/stars/nala-cub/prost?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nala-cub/prost)

- **[PandaLM_testset](https://github.com/WeOpenML/PandaLM)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/WeOpenML/PandaLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/WeOpenML/PandaLM)

- **[PromptBench](https://github.com/microsoft/promptbench)** - Tasks: `Sentiment Analysis`, `Grammar Correction`, `Natural Language Inference` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/microsoft/promptbench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/microsoft/promptbench)

- **[PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/michael-wzhu/PromptCBLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/michael-wzhu/PromptCBLUE)

- **[PubMedQA](https://huggingface.co/datasets/pubmed_qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fpubmed_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/pubmed_qa) [![GitHub Stars](https://img.shields.io/github/stars/pubmedqa/pubmedqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/pubmedqa/pubmedqa)

- **[QASC](https://huggingface.co/datasets/qasc)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fqasc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/qasc)

- **[QED](https://github.com/google-research-datasets/QED)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/QED?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/QED)

- **[QQP](https://huggingface.co/datasets/merve/qqp)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmerve%2Fqqp&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/merve/qqp)

- **[QiZhenGPT_eval](https://github.com/CMKRG/QiZhenGPT)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CMKRG/QiZhenGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CMKRG/QiZhenGPT)

- **[QuaRTz](https://huggingface.co/datasets/quartz)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fquartz&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/quartz)

- **[RACE](https://huggingface.co/datasets/race)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Frace&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/race)

- **[RAG-Instruct-Benchmark-Tester](https://huggingface.co/datasets/llmware/rag_instruct_benchmark_tester)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fllmware%2Frag_instruct_benchmark_tester&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/llmware/rag_instruct_benchmark_tester)

- **[RAGEval](https://github.com/OpenBMB/RAGEval)** - Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/OpenBMB/RAGEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenBMB/RAGEval)

- **[RGB](https://github.com/chen700564/RGB)** - Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/chen700564/RGB?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/chen700564/RGB)

- **[ROPES](https://huggingface.co/datasets/ropes)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fropes&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ropes)

- **[RTE](https://huggingface.co/datasets/glue/viewer/rte/train)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fglue%2Fviewer&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/glue/viewer/rte/train)

- **[RealTime QA](https://github.com/realtimeqa/realtimeqa_public)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/realtimeqa/realtimeqa_public?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/realtimeqa/realtimeqa_public)

- **[SCIBENCH](https://github.com/mandyyyyii/scibench)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/mandyyyyii/scibench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mandyyyyii/scibench)

- **[SNLI](https://huggingface.co/datasets/snli)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsnli&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/snli)

- **[SQuAD](https://huggingface.co/datasets/squad)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsquad&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/squad)

- **[SST-2](https://huggingface.co/datasets/sst2)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsst2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/sst2)

- **[STSB](https://huggingface.co/datasets/stsb_multi_mt)** - Tasks: `Semantic Similarity` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstsb_multi_mt&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stsb_multi_mt) [![GitHub Stars](https://img.shields.io/github/stars/PhilipMay/stsb-multi-mt?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/PhilipMay/stsb-multi-mt)

- **[SVAMP](https://github.com/arkilpatel/SVAMP)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/arkilpatel/SVAMP?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/arkilpatel/SVAMP)

- **[Safety Prompt](https://github.com/thu-coai/Safety-Prompts)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/thu-coai/Safety-Prompts?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-coai/Safety-Prompts)

- **[Safety-Prompts](https://github.com/thu-coai/Safety-Prompts)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/thu-coai/Safety-Prompts?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-coai/Safety-Prompts)

- **[SafetyBench](https://huggingface.co/datasets/thu-coai/SafetyBench)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fthu-coai%2FSafetyBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/thu-coai/SafetyBench) [![GitHub Stars](https://img.shields.io/github/stars/thu-coai/SafetyBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-coai/SafetyBench)

- **[SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhicai-zju%2FSciKnowEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hicai-zju/SciKnowEval) [![GitHub Stars](https://img.shields.io/github/stars/hicai-zju/sciknoweval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hicai-zju/sciknoweval)

- **[SciQ](https://huggingface.co/datasets/sciq)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsciq&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/sciq)

- **[ScienceQA](https://github.com/lupantech/ScienceQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/lupantech/ScienceQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lupantech/ScienceQA)

- **[SentEval](https://github.com/facebookresearch/SentEval)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/SentEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/SentEval)

- **[SocKET](https://github.com/minjechoi/SOCKET)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/minjechoi/SOCKET?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/minjechoi/SOCKET)

- **[Social IQa](https://huggingface.co/datasets/social_i_qa)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsocial_i_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/social_i_qa)

- **[StoryCloze](https://huggingface.co/datasets/story_cloze)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstory_cloze&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/story_cloze)

- **[SuperCLUE-Safety](https://github.com/CLUEbenchmark/SuperCLUE-safety)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/SuperCLUE-safety?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/SuperCLUE-safety)

- **[SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fm-a-p%2FSuperGPQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/m-a-p/SuperGPQA) [![GitHub Stars](https://img.shields.io/github/stars/SuperGPQA/SuperGPQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SuperGPQA/SuperGPQA)

- **[TRUSTGPT](https://github.com/HowieHwong/TrustGPT)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/HowieHwong/TrustGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HowieHwong/TrustGPT)

- **[TableBench](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMultilingual-Multimodal-NLP%2FTableBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench) [![GitHub Stars](https://img.shields.io/github/stars/TableBench/TableBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TableBench/TableBench)

- **[ToolEyes](https://github.com/Junjie-Ye/ToolEyes)** - Tasks: `Information Retrieval`, `Text Generation` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Junjie-Ye/ToolEyes?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Junjie-Ye/ToolEyes)

- **[TriviaQA](https://github.com/mandarjoshi90/triviaqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/mandarjoshi90/triviaqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mandarjoshi90/triviaqa)

- **[TruthfulQA](https://github.com/sylinrl/TruthfulQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/sylinrl/TruthfulQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sylinrl/TruthfulQA)

- **[TyDiQA](https://huggingface.co/datasets/tydiqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftydiqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tydiqa) [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/tydiqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/tydiqa)

- **[UHGEval](https://github.com/IAAR-Shanghai/UHGEval)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/IAAR-Shanghai/UHGEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/IAAR-Shanghai/UHGEval)

- **[WANLI](https://huggingface.co/datasets/alisawuffles/WANLI)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Falisawuffles%2FWANLI&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/alisawuffles/WANLI)

- **[WIQA](https://huggingface.co/datasets/wiqa)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwiqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/wiqa)

- **[WSC](https://huggingface.co/datasets/winograd_wsc/viewer/wsc273/test?row=0)** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwinograd_wsc%2Fviewer&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/winograd_wsc/viewer/wsc273/test?row=0)

- **[WYWEB](https://github.com/baudzhou/WYWEB)** - Tasks: `Sequence Labeling`, `Sentence Classification`, `Token Similarity`, `Reading Comprehension`, `Translation` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/baudzhou/WYWEB?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/baudzhou/WYWEB)

- **[WebQuestions](https://huggingface.co/datasets/web_questions)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fweb_questions&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/web_questions)

- **[WenMind 2024-5](https://github.com/SCUT-DLVCLab/WenMind)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/SCUT-DLVCLab/WenMind?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SCUT-DLVCLab/WenMind)

- **[WikiEval](https://huggingface.co/datasets/explodinggradients/WikiEval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fexplodinggradients%2FWikiEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/explodinggradients/WikiEval) [![GitHub Stars](https://img.shields.io/github/stars/explodinggradients/ragas?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/explodinggradients/ragas)

- **[WikiLingua](https://github.com/esdurmus/Wikilingua)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/esdurmus/Wikilingua?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/esdurmus/Wikilingua)

- **[WikiQA](https://huggingface.co/datasets/wiki_qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwiki_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/wiki_qa)

- **[WinoGrande](https://huggingface.co/datasets/winogrande)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwinogrande&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/winogrande) [![GitHub Stars](https://img.shields.io/github/stars/allenai/winogrande?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/winogrande)

- **[WinoWhy](https://github.com/HKUST-KnowComp/WinoWhy)** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/HKUST-KnowComp/WinoWhy?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HKUST-KnowComp/WinoWhy)

- **[WritingBench](https://github.com/X-PLUG/WritingBench)** - Tasks: `Instruction-Following`, `Summarization` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/X-PLUG/WritingBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/X-PLUG/WritingBench)

- **[XNLI](https://github.com/facebookresearch/XNLI)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English`, `French`, `Spanish`, `German`, `Chinese`, `Russian`, `Arabic`, `Hindi`, `Portuguese`, `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/XNLI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/XNLI)

- **[XSum](https://huggingface.co/datasets/EdinburghNLP/xsum)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FEdinburghNLP%2Fxsum&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/EdinburghNLP/xsum) [![GitHub Stars](https://img.shields.io/github/stars/EdinburghNLP/XSum?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/EdinburghNLP/XSum)

- **[XTREME](https://github.com/google-research-get/xtreme)** - Tasks: `Question Answering`, `Text Classification`, `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-get/xtreme?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-get/xtreme)

- **[XiezhiBenchmark](https://github.com/mikegu721/xiezhibenchmark)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/mikegu721/xiezhibenchmark?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mikegu721/xiezhibenchmark)

- **[YACLC](https://github.com/blcuicall/YACLC)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/blcuicall/YACLC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/blcuicall/YACLC)

- **[Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmultimodal-reasoning-lab%2FZebra-CoT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) [![GitHub Stars](https://img.shields.io/github/stars/multimodal-reasoning-lab/Bagel-Zebra-CoT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT)

- **[ZebraLogic](https://huggingface.co/datasets/WildEval/ZebraLogic)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FWildEval%2FZebraLogic&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/WildEval/ZebraLogic) [![GitHub Stars](https://img.shields.io/github/stars/WildEval/ZeroEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/WildEval/ZeroEval)

- **[aclue](https://huggingface.co/datasets/tyouisen/aclue)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftyouisen%2Faclue&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tyouisen/aclue) [![GitHub Stars](https://img.shields.io/github/stars/isen-zhang/aclue?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/isen-zhang/aclue)

- **[ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fceval%2Fceval-exam&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ceval/ceval-exam) [![GitHub Stars](https://img.shields.io/github/stars/SJTU-LIT/ceval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SJTU-LIT/ceval)

- **[cmath](https://huggingface.co/datasets/weitianwen/cmath)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fweitianwen%2Fcmath&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/weitianwen/cmath) [![GitHub Stars](https://img.shields.io/github/stars/XiaoMi/cmath?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/XiaoMi/cmath)

- **[cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhaonan-li%2Fcmmlu&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/haonan-li/cmmlu) [![GitHub Stars](https://img.shields.io/github/stars/haonan-li/CMMLU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/haonan-li/CMMLU)

- **[decaNLP](https://github.com/salesforce/decaNLP)** - Tasks: `Question Answering`, `Machine Translation`, `Summarization`, `Natural Language Inference` | Mod: `Text` | Lang: `English`, `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/salesforce/decaNLP?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/salesforce/decaNLP)

- **[gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FIdavidrein%2Fgpqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Idavidrein/gpqa) [![GitHub Stars](https://img.shields.io/github/stars/idavidrein/gpqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/idavidrein/gpqa)

- **[healthsearchqa](https://huggingface.co/datasets/katielink/healthsearchqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fkatielink%2Fhealthsearchqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/katielink/healthsearchqa)

- **[huatuo26M-testdatasets](https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFreedomIntelligence%2Fhuatuo26M-testdatasets&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets) [![GitHub Stars](https://img.shields.io/github/stars/FreedomIntelligence/Huatuo-26M?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FreedomIntelligence/Huatuo-26M)

- **[kobest_v1](https://huggingface.co/datasets/skt/kobest_v1)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Korean` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fskt%2Fkobest_v1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/skt/kobest_v1)

- **[legalbench](https://huggingface.co/datasets/nguha/legalbench)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnguha%2Flegalbench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nguha/legalbench) [![GitHub Stars](https://img.shields.io/github/stars/HazyResearch/legalbench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HazyResearch/legalbench)

- **[lila](https://huggingface.co/datasets/allenai/lila)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Flila&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/lila) [![GitHub Stars](https://img.shields.io/github/stars/allenai/Lila?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/Lila)

- **[mmlu-redux-2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fedinburgh-dawg%2Fmmlu-redux-2.0&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0) [![GitHub Stars](https://img.shields.io/github/stars/aryopg/mmlu-redux?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/aryopg/mmlu-redux)

- **[mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Fmt_bench_human_judgments&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) [![GitHub Stars](https://img.shields.io/github/stars/lm-sys/FastChat?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

- **[raft](https://huggingface.co/datasets/ought/raft)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fought%2Fraft&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ought/raft)

- **[tmmluplus](https://huggingface.co/datasets/ikala/tmmluplus)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fikala%2Ftmmluplus&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ikala/tmmluplus)

- **[zero_scrolls](https://huggingface.co/datasets/tau/zero_scrolls)** - Tasks: `Summarization`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftau%2Fzero_scrolls&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tau/zero_scrolls) [![GitHub Stars](https://img.shields.io/github/stars/tau-nlp/zero_scrolls?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tau-nlp/zero_scrolls)



<a id="text-retrieval-rag"></a>
#### Retrieval / RAG
- **[CRUD-RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/IAAR-Shanghai/CRUD_RAG?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/IAAR-Shanghai/CRUD_RAG)

- **[LFRQA](https://github.com/awslabs/rag-qa-arena)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/awslabs/rag-qa-arena?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/awslabs/rag-qa-arena)

- **[MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fyixuantt%2FMultiHopRAG&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/yixuantt/MultiHopRAG) [![GitHub Stars](https://img.shields.io/github/stars/yixuantt/MultiHop-RAG?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/yixuantt/MultiHop-RAG/)

---

<a id="code"></a>
### Code


<a id="code-pretraining"></a>
#### Pretraining
- **Github** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `English`, `Multi`

- **[CodeAlpaca‑20K](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsahil2801%2FCodeAlpaca-20k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) [![GitHub Stars](https://img.shields.io/github/stars/samuelblais/CodeAlpaca-20K?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/samuelblais/CodeAlpaca-20K)

- **[CodeParrot](https://huggingface.co/datasets/codeparrot/codeparrot-clean)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcodeparrot%2Fcodeparrot-clean&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/codeparrot/codeparrot-clean) [![GitHub Stars](https://img.shields.io/github/stars/bigcode-project/CodeParrot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/bigcode-project/CodeParrot)

- **[starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigcode%2Fstarcoderdata&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigcode/starcoderdata) [![GitHub Stars](https://img.shields.io/github/stars/huggingface/starcoderdata?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/huggingface/starcoderdata)

- **[the-stack](https://huggingface.co/datasets/bigcode/the-stack)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `English`, `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigcode%2Fthe-stack&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigcode/the-stack)



<a id="code-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[Code_Alpaca_20K](https://github.com/sahil280114/codealpaca)** - Tasks: `Code Generation`, `Code Completion`, `Code Summarization` | Mod: `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/sahil280114/codealpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sahil280114/codealpaca)

- **[MBPP](https://github.com/google-research/google-research/tree/master/mbpp)** - Tasks: `Code Generation`, `Code Completion` | Mod: `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research/google-research?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research/google-research/tree/master/mbpp)

- **[Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)** - Tasks: `Instruction-Following`, `Code Generation`, `Code Completion` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fise-uiuc%2FMagicoder-OSS-Instruct-75K&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) [![GitHub Stars](https://img.shields.io/github/stars/ise-uiuc/magicoder?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ise-uiuc/magicoder)

- **[function- invocations-25k](https://huggingface.co/datasets/unaidedelf87777/openapi-function-invocations-25k)** - Tasks: `Instruction-Following`, `Tool-Use` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Funaidedelf87777%2Fopenapi-function-invocations-25k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/unaidedelf87777/openapi-function-invocations-25k) [![GitHub Stars](https://img.shields.io/github/stars/APIs-guru/openapi-directory?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/APIs-guru/openapi-directory)

- **[instructional_ codesearchnet_python](https://huggingface.co/datasets/Nan-Do/instructional_codesearchnet_python)** - Tasks: `Instruction-Following`, `Code Generation` | Mod: `Code` | Lang: `English`, `Python` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FNan-Do%2Finstructional_codesearchnet_python&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Nan-Do/instructional_codesearchnet_python)

- **[openapi-function-invocations‑25k](https://huggingface.co/datasets/unaidedelf87777/openapi-function-invocation-25k)** - Tasks: `Code Generation`, `Code Completion` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Funaidedelf87777%2Fopenapi-function-invocation-25k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/unaidedelf87777/openapi-function-invocation-25k) [![GitHub Stars](https://img.shields.io/github/stars/openai/openapi-function-invocations-25k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/openapi-function-invocations-25k)


<a id="code-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*


<a id="code-evaluation-benchmark"></a>
#### Evaluation / Benchmark

- **[APIBench](https://huggingface.co/datasets/gorilla-llm/APIBench)** - Tasks: `Reasoning`, `Tool-Use` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fgorilla-llm%2FAPIBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/gorilla-llm/APIBench) [![GitHub Stars](https://img.shields.io/github/stars/ShishirPatil/gorilla?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ShishirPatil/gorilla)

- **[Berkeley Function Calling Leaderboard (BFCL)](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)** - Tasks: `Tool-Use` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fgorilla-llm%2FBerkeley-Function-Calling-Leaderboard&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)

- **[CodeElo](https://huggingface.co/datasets/Qwen/CodeElo)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FQwen%2FCodeElo&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Qwen/CodeElo) [![GitHub Stars](https://img.shields.io/github/stars/QwenLM/CodeElo?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/QwenLM/CodeElo)

- **[DS-1000](https://github.com/xlang-ai/DS-1000)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/xlang-ai/DS-1000?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/xlang-ai/DS-1000)

- **[DomainEval 2024-8](https://github.com/domaineval/DomainEval)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/domaineval/DomainEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/domaineval/DomainEval)

- **[HumanEval+ 2023-5](https://github.com/evalplus/evalplus)** - Tasks: `Code Generation`, `Code Repair` | Mod: `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/evalplus/evalplus?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/evalplus/evalplus)

- **[HumanEval](https://github.com/openai/human-eval)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/openai/human-eval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/human-eval)

- **[MTPB](https://github.com/salesforce/CodeGen)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/salesforce/CodeGen?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/salesforce/CodeGen)

- **[apps](https://huggingface.co/datasets/codeparrot/apps)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcodeparrot%2Fapps&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/codeparrot/apps) [![GitHub Stars](https://img.shields.io/github/stars/hendrycks/apps?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hendrycks/apps)

- **[cruxeval](https://huggingface.co/datasets/cruxeval-org/cruxeval)** - Tasks: `Reasoning` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcruxeval-org%2Fcruxeval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cruxeval-org/cruxeval) [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/cruxeval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/cruxeval)

- **[humanevalpack](https://huggingface.co/datasets/bigcode/humanevalpack)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigcode%2Fhumanevalpack&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigcode/humanevalpack) [![GitHub Stars](https://img.shields.io/github/stars/bigcode-project/octopack?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/bigcode-project/octopack)


<a id="code-retrieval-rag"></a>
#### Retrieval / RAG
- *(add entries)*

---

<a id="multimodal"></a>
### Multimodal


<a id="multimodal-pretraining"></a>
#### Pretraining
- **ROOTS** - Tasks: `Machine Translation`, `Text Classification`, `Information Extraction` | Mod: `Text`, `Code` | Lang: `Multi`

- **[OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)** - Tasks: `Image Captioning`, `VQA`, `Information Extraction` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceM4%2FOBELICS&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)

- **[OBELISC](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)** - Mod: `Text`, `Image` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceM4%2FOBELICS&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) [![GitHub Stars](https://img.shields.io/github/stars/huggingface/OBELICS?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/huggingface/OBELICS)

- **[OIG‑43M](https://huggingface.co/datasets/laion/OIG)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering`, `Machine Translation` | Mod: `Text`, `Image` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flaion%2FOIG&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/laion/OIG)

- **[The Pile (V1)](https://github.com/EleutherAI/the-pile)** - Mod: `Text`, `Code` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/EleutherAI/the-pile?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/EleutherAI/the-pile)

- **[xP3 (and some variant)](https://huggingface.co/datasets/bigscience/xP3)** - Tasks: `Instruction-Following` | Mod: `Text`, `Code` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigscience%2FxP3&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigscience/xP3)



<a id="multimodal-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[COIG](https://huggingface.co/datasets/BAAI/COIG)** - Tasks: `Question Answering`, `Code Generation` | Mod: `Text`, `Code` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBAAI%2FCOIG&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BAAI/COIG)

- **[JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB)** - Tasks: `Question Answering`, `Image Captioning` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FJourneyDB%2FJourneyDB&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/JourneyDB/JourneyDB)

- **[LLaVA Instruction](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)** - Tasks: `Instruction-Following`, `VQA` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fliuhaotian%2FLLaVA-Instruct-150K&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) [![GitHub Stars](https://img.shields.io/github/stars/haotian-liu/LLaVA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/haotian-liu/LLaVA)

- **[LLaVA Visual Instruct 150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)** - Tasks: `Instruction-Following` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fliuhaotian%2FLLaVA-Instruct-150K&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) [![GitHub Stars](https://img.shields.io/github/stars/haotian-liu/LLaVA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/haotian-liu/LLaVA)

- **[M3IT](https://huggingface.co/datasets/MMInstruction/M3IT)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMMInstruction%2FM3IT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MMInstruction/M3IT)

- **[MIMIC-IT](https://github.com/Luodian/Otter/tree/main/mimic-it)** - Tasks: `Instruction-Following`, `Video Understanding` | Mod: `Image`, `Video` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/Luodian/Otter?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Luodian/Otter/tree/main/mimic-it)

- **[VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text`, `Video` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenGVLab%2FVideoChat2-IT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)




<a id="multimodal-alignment-rlhf"></a>
#### Alignment / RL
- **[cc_sbu_align](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align)** - Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FVision-CAIR%2Fcc_sbu_align&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) [![GitHub Stars](https://img.shields.io/github/stars/haotian07/miniGPT-4?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/haotian07/miniGPT-4)




<a id="multimodal-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- **[ALM-Bench](https://huggingface.co/datasets/MBZUAI/ALM-Bench)** - Mod: `Text`, `Image`, `Video`, `Audio` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMBZUAI%2FALM-Bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MBZUAI/ALM-Bench) [![GitHub Stars](https://img.shields.io/github/stars/mbzuai-oryx/ALM-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mbzuai-oryx/ALM-Bench)

- **[CodeXGLUE](https://github.com/microsoft/CodeXGLUE)** - Tasks: `Code Generation`, `Code Completion`, `Code Summarization` | Mod: `Code`, `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/microsoft/CodeXGLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/microsoft/CodeXGLUE)

- **[II-Bench](https://huggingface.co/datasets/m-a-p/II-Bench)** - Mod: `Image` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fm-a-p%2FII-Bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/m-a-p/II-Bench) [![GitHub Stars](https://img.shields.io/github/stars/II-Bench/II-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/II-Bench/II-Bench)

- **[MMIU](https://huggingface.co/datasets/FanqingM/MMIU-Benchmark)** - Tasks: `Image Understanding` | Mod: `Image` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFanqingM%2FMMIU-Benchmark&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/FanqingM/MMIU-Benchmark) [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/MMIU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenGVLab/MMIU)

- **[MRAG-Bench](https://huggingface.co/datasets/uclanlp/MRAG-Bench)** | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fuclanlp%2FMRAG-Bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/uclanlp/MRAG-Bench) [![GitHub Stars](https://img.shields.io/github/stars/mragbench/MRAG-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mragbench/MRAG-Bench)

- **[MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench)** - Tasks: `Video Understanding` | Mod: `Video` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenGVLab%2FMVBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenGVLab/MVBench) [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/Ask-Anything?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)

- **[MedTrinity-25M](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text`, `Image` | Lang: `English`, `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FUCSC-VLAA%2FMedTrinity-25M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M) [![GitHub Stars](https://img.shields.io/github/stars/UCSC-VLAA/MedTrinity-25M?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/UCSC-VLAA/MedTrinity-25M)

- **[ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)** - Tasks: `Image Captioning` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FLin-Chen%2FShareGPT4V&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)

- **[TabMWP](https://github.com/lupantech/PromptPG)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text`, `Table` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/lupantech/PromptPG?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lupantech/PromptPG)



<a id="multimodal-retrieval-rag"></a>
#### Retrieval / RAG
- **ViDoRe** - Tasks: `Question Answering`, `VQA` | Mod: `Text`, `Image`, `Video` | Lang: `English`

- **[M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR)** | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTIGER-Lab%2FM-BEIR&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) [![GitHub Stars](https://img.shields.io/github/stars/TIGER-AI-Lab/UniIR?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TIGER-AI-Lab/UniIR)

- **[M2KR](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering)** | [![GitHub Stars](https://img.shields.io/github/stars/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering)


---

<a id="gen"></a>
### Generation (Image/Video/Audio)


<a id="gen-pretraining"></a>
#### Pretraining
- **[Dataset-E](link)** — Tags: `GeneralLM`, `Image-Text`, `English` — Pretraining for image generation…


<a id="gen-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[WebNLG](https://huggingface.co/datasets/web_nlg)** - Tasks: `Text Generation` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fweb_nlg&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/web_nlg) [![GitHub Stars](https://img.shields.io/github/stars/fuzihaofzh/webnlg-dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/fuzihaofzh/webnlg-dataset)


<a id="gen-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*


<a id="gen-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- **[ODEX](https://github.com/zorazrw/odex)** - Tasks: `Code Generation` | Mod: `Text`, `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/zorazrw/odex?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/zorazrw/odex)



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
- **[MINT](https://github.com/xingyaoww/mint-bench)** - Tasks: `Tool-Use`, `Reasoning` | Mod: `Text`, `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/xingyaoww/mint-bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/xingyaoww/mint-bench)

- **[xLAM Function Calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)** - Tasks: `Tool-Use` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSalesforce%2Fxlam-function-calling-60k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
<a id="agent-alignment-rlhf"></a>


<a id="agent-instruction-tuning"></a>
#### Alignment / RL
- *(add entries)*


<a id="agent-evaluation-benchmark"></a>
#### Evaluation / Benchmark

- **[API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)** - Tasks: `Dialogue`, `Question Answering`, `Tool-Use`, `Planning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/AlibabaResearch/DAMO-ConvAI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)

- **[AgentBench](https://github.com/THUDM/AgentBench)** - Tasks: `Reasoning`, `Planning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/THUDM/AgentBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/AgentBench)

- **[GameBench 2024-6](https://github.com/Joshuaclymer/GameBench)** - Tasks: `Reasoning`, `Planning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Joshuaclymer/GameBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Joshuaclymer/GameBench)

- **[LangChainDatasets](https://github.com/hwchase17/langchain-datasets)** - Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/hwchase17/langchain-datasets?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hwchase17/langchain-datasets)

- **[ParlAI](https://github.com/facebookresearch/ParlAI)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/ParlAI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/ParlAI)

- **[SuperCLUE-Agent](https://github.com/CLUEbenchmark/SuperCLUE-Agent)** - Tasks: `Tool-Use`, `Planning` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/SuperCLUE-Agent?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/SuperCLUE-Agent)

- **[ToolBench](https://github.com/sambanova/toolbench)** - Tasks: `Tool-Use`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/sambanova/toolbench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sambanova/toolbench)


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


