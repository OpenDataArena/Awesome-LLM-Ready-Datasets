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
- **[AmericanStories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories)** - Tasks: `Text Classification`, `Information Extraction`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdell-research-harvard%2FAmericanStories&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/dell-research-harvard/AmericanStories) [![GitHub Stars](https://img.shields.io/github/stars/dell-research-harvard/AmericanStories?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/dell-research-harvard/AmericanStories)   

- **[ArabicText 2022](https://data.baai.ac.cn/datadetail/ArabicText-2022)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Arabic`    

- **[arXiv-papers](https://github.com/mattbierbaum/arxiv-public-datasets)** - Tasks: `Text Classification`, `Information Extraction`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnick007x%2Farxiv-papers&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nick007x/arxiv-papers)   

- **[arXiv-public-datasets](https://github.com/mattbierbaum/arxiv-public-datasets)** - Tasks: `Text Classification`, `Information Extraction`, `Summarization` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/mattbierbaum/arxiv-public-datasets?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mattbierbaum/arxiv-public-datasets)   

- **[awesome chinese legal resources](https://github.com/pengxiao-song/awesome-chinese-legal-resources)** - Tasks: `Question Answering`, `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/pengxiao-song/awesome-chinese-legal-resources?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/pengxiao-song/awesome-chinese-legal-resources)   

- **[Baidu baike](https://github.com/BIT-ENGD/baidu_baike)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/BIT-ENGD/baidu_baike?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/BIT-ENGD/baidu_baike)    

- **[BNC](https://www.natcorp.ox.ac.uk/)** - Tasks: `Text Classification`, `Information Extraction` and so on | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSoranBD%2FBNC&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/SoranBD/BNC) [![GitHub Stars](https://img.shields.io/github/stars/katsonoda/BNC-XML?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/katsonoda/BNC-XML)    

- **[bookcorpusopen](https://github.com/jackbandy/bookcorpus-datasheet)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` |  [![GitHub Stars](https://img.shields.io/github/stars/jackbandy/bookcorpus-datasheet?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jackbandy/bookcorpus-datasheet)   

- **[c4](https://huggingface.co/datasets/allenai/c4)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fc4&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/c4)   

- **[CBook-150K](https://github.com/FudanNLPLAB/CBook-150K)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/FudanNLPLAB/CBook-150K?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FudanNLPLAB/CBook-150K)   

- **[cc100](https://huggingface.co/datasets/statmt/cc100)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Multi`  | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstatmt%2Fcc100&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/statmt/cc100)   

- **[ChineseWebText](https://huggingface.co/datasets/CASIA-LM/ChineseWebText)** - Tasks: `Information Extraction` ｜ Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FCASIA-LM%2FChineseWebText&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/CASIA-LM/ChineseWebText) [![GitHub Stars](https://img.shields.io/github/stars/CASIA-LM/ChineseWebText?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CASIA-LM/ChineseWebText)    

- **[ChineseWebText2.0](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0)** - Tasks: `Information Extraction` ｜Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FCASIA-LM%2FChineseWebText2.0&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0) [![GitHub Stars](https://img.shields.io/github/stars/CASIA-LM/ChineseWebText-2.0?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CASIA-LM/ChineseWebText-2.0)    

- **[CLUECorpus](https://github.com/CLUEbenchmark/CLUE)** - Tasks: `Information Extraction`, `Question Answering` ｜Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUE)   

- **[CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)** - Tasks: `Information Extraction`, `Question Answering` ｜ Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUECorpus2020?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUECorpus2020)    

- **[Common Crawl](https://github.com/facebookresearch/cc_net)** - Tasks: `Information Extraction` ｜ Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/cc_net?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/cc_net)    

- **[CSL](https://github.com/ydli-ai/CSL)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/ydli-ai/CSL?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ydli-ai/CSL)   

- **[CSLDCP](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/csldcp)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/FewCLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/csldcp)   

- **[dolma](https://huggingface.co/datasets/allenai/dolma)** - Tasks: `Question Answering`, `Summarization` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fdolma&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/dolma)   

- **[Expository-Prose-V1](https://huggingface.co/datasets/pints-ai/Expository-Prose-V1)** - Tasks: `Text Classification`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fpints-ai%2FExpository-Prose-V1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/pints-ai/Expository-Prose-V1) [![GitHub Stars](https://img.shields.io/github/stars/Pints-AI/1.5-Pints?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Pints-AI/1.5-Pints)    

- **[falcon-refinedweb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftiiuae%2Ffalcon-refinedweb&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)   

- **[FinNLP](https://github.com/AI4Finance-Foundation/FinNLP)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinNLP?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AI4Finance-Foundation/FinNLP)    

- **[Future-Idea-Generation](https://github.com/sandeep82945/Future-Idea-Generation)** - Tasks: `Instruction-Following`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/sandeep82945/Future-Idea-Generation?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sandeep82945/Future-Idea-Generation)    

- **[Gutenberg project](https://huggingface.co/datasets/manu/project_gutenberg)** - Tasks: `Text Classification`, `Summarization` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmanu%2Fproject_gutenberg&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/manu/project_gutenberg) [![GitHub Stars](https://img.shields.io/github/stars/WordPress/gutenberg?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/WordPress/gutenberg)   

- **[mc4](https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned)** - Tasks: `Text Classification`, `Information Extraction`, `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fyhavinga%2Fmc4_nl_cleaned&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned)    

- **[mOSCAR](https://huggingface.co/datasets/oscar-corpus/mOSCAR)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Foscar-corpus%2FmOSCAR&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/oscar-corpus/mOSCAR)   

- **[MultiUN](https://huggingface.co/datasets/Helsinki-NLP/multiun)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHelsinki-NLP%2Fmultiun&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Helsinki-NLP/multiun)   

- **[nlp_Chinese_Corpus](https://github.com/brightmart/nlp_chinese_corpus)** - Tasks: `Question Answering`, `Summarization` | Mod: `Text` | Lang: `Chinese`, `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/brightmart/nlp_chinese_corpus?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/brightmart/nlp_chinese_corpus)   

- **[open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopen-web-math%2Fopen-web-math&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/open-web-math/open-web-math) [![GitHub Stars](https://img.shields.io/github/stars/keirp/OpenWebMath?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/keirp/OpenWebMath)   

- **[OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)** - Tasks: `Text Classification`, `Summarization`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSkylion007%2Fopenwebtext&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Skylion007/openwebtext) [![GitHub Stars](https://img.shields.io/github/stars/jcpeterson/openwebtext?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jcpeterson/openwebtext)   

- **[OSCAR-2201](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201)** - Tasks: `Summarization`  | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Foscar-corpus%2FOSCAR-2201&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201)   

- **[paracrawl_context](https://huggingface.co/datasets/Proyag/paracrawl_context)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FProyag%2Fparacrawl_context&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Proyag/paracrawl_context) [![GitHub Stars](https://img.shields.io/github/stars/Proyag/ParaCrawl-Context?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Proyag/ParaCrawl-Context)   

- **[peS2o](https://huggingface.co/datasets/allenai/peS2o)** - Tasks: `Question Answering`, `Summarization`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2FpeS2o&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/peS2o)   

- **[pretrain_en](https://huggingface.co/datasets/TigerResearch/pretrain_en)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Fpretrain_en&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/pretrain_en) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)   

- **[pretrain_zh](https://huggingface.co/datasets/TigerResearch/pretrain_zh)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Fpretrain_zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/pretrain_zh) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)   

- **[proof-pile](https://huggingface.co/datasets/hoskinson-center/proof-pile)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhoskinson-center%2Fproof-pile&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hoskinson-center/proof-pile)   

- **[PubMed Central](https://huggingface.co/datasets/pmc/open_access)** - Tasks: `Information Extraction`, `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fpmc%2Fopen_access&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/pmc/open_access)   

- **[Pushshift Reddit](https://huggingface.co/datasets/fddemarco/pushshift-reddit)** - Tasks: `Dialogue`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ffddemarco%2Fpushshift-reddit&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/fddemarco/pushshift-reddit)   

- **[RealNews](https://github.com/rowanz/grover)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/rowanz/grover?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/rowanz/grover)   

- **[Reddit](https://github.com/webis-de/webis-tldr-17-corpus)** - Tasks: `Dialogue`, `Information Extraction`, `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/webis-de/webis-tldr-17-corpus?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/webis-de/webis-tldr-17-corpus)     

- **[RedPajama-Data-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftogethercomputer%2FRedPajama-Data-1T&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) [![GitHub Stars](https://img.shields.io/github/stars/togethercomputer/RedPajama-Data?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/togethercomputer/RedPajama-Data)   

- **[RedPajama-Data-V2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftogethercomputer%2FRedPajama-Data-V2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) [![GitHub Stars](https://img.shields.io/github/stars/togethercomputer/RedPajama-Data?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/togethercomputer/RedPajama-Data)   

- **[S2ORC](https://github.com/allenai/s2orc)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/allenai/s2orc?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/s2orc)   

- **[SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcerebras%2FSlimPajama-627B&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cerebras/SlimPajama-627B)   

- **[StackExchange](https://huggingface.co/datasets/teven/stackexchange)** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fteven%2Fstackexchange&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/teven/stackexchange)   

- **[The Pile](https://github.com/EleutherAI/the-pile)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering`, `Reasoning`, `Summarization` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/EleutherAI/the-pile?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/EleutherAI/the-pile)   

- **[TigerBot Series](https://github.com/TigerResearch/TigerBot#%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%E9%9B%86)** - Tasks: `Text Classification`, `Information Extraction`, `Machine Translation` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot#%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%E9%9B%86)   

- **[Toronto Book Corpus](https://github.com/sgraaf/Replicate-Toronto-BookCorpus)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/sgraaf/Replicate-Toronto-BookCorpus?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sgraaf/Replicate-Toronto-BookCorpus)   

- **[UNCorpus v1.0](https://huggingface.co/datasets/wmt/uncorpus)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwmt%2Funcorpus&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/wmt/uncorpus/tree/main)   

- **[WebText](https://github.com/openai/gpt-2)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/openai/gpt-2?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/gpt-2)   

- **[WikiMatrix](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/LASER?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/LASER/tree/main/tasks/WikiMatrix)   

- **[wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)** - Tasks: `Question Answering`, `Summarization`, `Text Classification` | Mod: `Text` | Lang: `English`, `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwikipedia&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/wikimedia/wikipedia)   

- **[WuDaoCorpora-Text](https://www.scidb.cn/en/detail?dataSetId=c6a3fe684227415a9db8e21bac4a15ab)** - Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmdokl%2FWuDaoCorpora2.0-RefinedEdition60GTXT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mdokl/WuDaoCorpora2.0-RefinedEdition60GTXT)   

- **[Zhihu](https://huggingface.co/datasets/suolyer/zhihu)** - Tasks: `Dialogue`, `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsuolyer%2Fzhihu&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/suolyer/zhihu)   



<a id="text-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[0.5M version](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Ftrain_0.5M_CN&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)[![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE?tab=readme-ov-file)  

- **[Adversarial QA](https://huggingface.co/datasets/UCLNLP/adversarial_qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FUCLNLP%2Fadversarial_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/UCLNLP/adversarial_qa) [![GitHub Stars](https://img.shields.io/github/stars/maxbartolo/adversarialQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/maxbartolo/adversarialQA)  

- **[AESLC](https://huggingface.co/datasets/Yale-LILY/aeslc)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fquoref&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/quoref)[![GitHub Stars](https://img.shields.io/github/stars/ryanzhumich/AESLC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ryanzhumich/AESLC)  

- **[AGNEWS](https://huggingface.co/datasets/fancyzhx/ag_news)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ffancyzhx%2Fag_news&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/fancyzhx/ag_news)   

- **[ALLaVA-4V Data](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFreedomIntelligence%2FALLaVA-4V&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) [![GitHub Stars](https://img.shields.io/github/stars/FreedomIntelligence/ALLaVA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FreedomIntelligence/ALLaVA)  

- **[Alpaca Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftatsu-lab%2Falpaca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tatsu-lab/alpaca) [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca)  

- **[Alpaca GPT‑4 Chinese](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fllamafactory%2Falpaca_gpt4_zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/llamafactory/alpaca_gpt4_zh)  

- **[Alpaca GPT‑4 Data](https://huggingface.co/datasets/vicgalle/alpaca-gpt4)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fvicgalle%2Falpaca-gpt4&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca)  

- **[alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fshibing624%2Falpaca-zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shibing624/alpaca-zh) [![GitHub Stars](https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#data-release)   

- **[AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/gururise/AlpacaDataCleaned?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/gururise/AlpacaDataCleaned)[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Falexl83%2FAlpacaDataCleaned&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/alexl83/AlpacaDataCleaned)  

- **[Alpaca‑CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)** - Tasks: `Instruction-Following`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FQingyiSi%2FAlpaca-CoT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) [![GitHub Stars](https://img.shields.io/github/stars/tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tatsu-lab/stanford_alpaca)  

- **[Ape210K](https://github.com/Chenny0808/ape210k)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Chenny0808/ape210k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Chenny0808/ape210k)  

- **[AQUA-RAT](https://huggingface.co/datasets/deepmind/aqua_rat)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdeepmind%2Faqua_rat&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/deepmind/aqua_rat) [![GitHub Stars](https://img.shields.io/github/stars/google-deepmind/AQuA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-deepmind/AQuA)  

- **[arxiv‑math‑instruct‑50k](https://huggingface.co/datasets/Sharathhebbar24/arxiv-math-instruct-50k)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSharathhebbar24%2Farxiv-math-instruct-50k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Sharathhebbar24/arxiv-math-instruct-50k)   

- **[ASDiv](https://huggingface.co/datasets/EleutherAI/asdiv)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FEleutherAI%2Fasdiv&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/EleutherAI/asdiv) [![GitHub Stars](https://img.shields.io/github/stars/chaochun/nlu-asdiv-dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/chaochun/nlu-asdiv-dataset)  

- **[Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering`, `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMBZUAI%2FBactrian-X&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MBZUAI/Bactrian-X)[![GitHub Stars](https://img.shields.io/github/stars/mbzuai-nlp/Bactrian-X?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mbzuai-nlp/Bactrian-X)  

- **[Baize Dataset](https://huggingface.co/datasets/linkanjarad/baize-chat-data)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flinkanjarad%2Fbaize-chat-data&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/linkanjarad/baize-chat-data) [![GitHub Stars](https://img.shields.io/github/stars/project-baize/baize-chatbot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/project-baize/baize-chatbot/tree/main/data)  

- **[BELLE](https://github.com/LianjiaTech/BELLE)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE)  

- **[blended_skill_talk](https://huggingface.co/datasets/ParlAI/blended_skill_talk)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FParlAI%2Fblended_skill_talk&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ParlAI/blended_skill_talk) [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/ParlAI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/blended_skill_talk)   

- **[BQ](https://huggingface.co/datasets/shibing624/nli_zh)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fshibing624%2Fnli_zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shibing624/nli_zh)  

- **[C3](https://huggingface.co/datasets/dataset-org/c3)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdataset-org%2Fc3&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/dataset-org/c3) [![GitHub Stars](https://img.shields.io/github/stars/nlpdata/c3?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nlpdata/c3)  

- **[Cabrita Dataset](https://github.com/22-hours/cabrita/tree/main/data)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Portuguese` | [![GitHub Stars](https://img.shields.io/github/stars/22-hours/cabrita?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/22-hours/cabrita/tree/main/data)   

- **[ChatAlpaca data](https://github.com/cascip/ChatAlpaca)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/cascip/ChatAlpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/cascip/ChatAlpaca)  

- **[Chatbot Arena Conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Fchatbot_arena_conversations&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)  

- **[ChatDoctor](https://github.com/Kent0n-Li/ChatDoctor)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Kent0n-Li/ChatDoctor?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Kent0n-Li/ChatDoctor)  

- **[Chatgpt_corpus](https://github.com/PlexPt/chatgpt-corpus/releases/tag/3)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/PlexPt/chatgpt-corpus?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/PlexPt/chatgpt-corpus/releases/tag/3)    

- **[ChatMed_Consult_Dataset](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmichaelwzhu%2FChatMed_Consult_Dataset&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset) [![GitHub Stars](https://img.shields.io/github/stars/michael-wzhu/ChatMed?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/michael-wzhu/ChatMed)  

- **[Child_chat_data](https://github.com/HIT-SCIR-SC/QiaoBan)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/HIT-SCIR-SC/QiaoBan?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HIT-SCIR-SC/QiaoBan)  

- **[CLiB](https://github.com/jeinlee1991/chinese-llm-benchmark)** - Tasks: `Classification`, `Information Extraction`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/jeinlee1991/chinese-llm-benchmark?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jeinlee1991/chinese-llm-benchmark)  

- **[CLOTH](https://huggingface.co/datasets/AndyChiang/cloth)** - Tasks: `Instruction-Following`, `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FAndyChiang%2Fcloth&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/AndyChiang/cloth)  

- **[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUENER2020?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUENER2020)  

- **[CMD](https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FticoAg%2FChinese-medical-dialogue&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ticoAg/Chinese-medical-dialogue) [![GitHub Stars](https://img.shields.io/github/stars/Toyhom/Chinese-medical-dialogue-data?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Toyhom/Chinese-medical-dialogue-data)  

- **[cMedQA2](https://huggingface.co/datasets/fzkuji/cMedQA2)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ffzkuji%2FcMedQA2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/fzkuji/cMedQA2) [![GitHub Stars](https://img.shields.io/github/stars/zhangsheng93/cMedQA2?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/zhangsheng93/cMedQA2)   

- **[CMRC2018](https://huggingface.co/datasets/hfl/cmrc2018)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhfl%2Fcmrc2018&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hfl/cmrc2018) [![GitHub Stars](https://img.shields.io/github/stars/ymcui/cmrc2018?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ymcui/cmrc2018)  

- **[CMtMedQA](https://huggingface.co/datasets/Suprit/CMtMedQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSuprit%2FCMtMedQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Suprit/CMtMedQA) [![GitHub Stars](https://img.shields.io/github/stars/SupritYoung/Zhongjing?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SupritYoung/Zhongjing)  

- **[CNewSum](https://huggingface.co/datasets/ethanhao2077/cnewsum-processed)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fethanhao2077%2Fcnewsum-processed&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ethanhao2077/cnewsum-processed) [![GitHub Stars](https://img.shields.io/github/stars/dqwang122/MLROUGE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/dqwang122/MLROUGE)  

- **[CNN-DM](https://huggingface.co/datasets/hoang1123/cnndm)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhoang1123%2Fcnndm&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hoang1123/cnndm)  

- **[COIG](https://huggingface.co/datasets/BAAI/COIG)** - Tasks: `Question Answering`, `Code Generation` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBAAI%2FCOIG&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BAAI/COIG) 

- **[CommonGen](https://huggingface.co/datasets/allenai/common_gen)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fcommon_gen&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/common_gen)[![GitHub Stars](https://img.shields.io/github/stars/INK-USC/CommonGen?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/INK-USC/CommonGen)  

- **[CoQA](https://huggingface.co/datasets/stanfordnlp/coqa)** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstanfordnlp%2Fcoqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stanfordnlp/coqa) [![GitHub Stars](https://img.shields.io/github/stars/stanfordnlp/coqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/stanfordnlp/coqa)  

- **[CrossWOZ](https://huggingface.co/datasets/GEM/CrossWOZ)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGEM%2FCrossWOZ&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/GEM/CrossWOZ) [![GitHub Stars](https://img.shields.io/github/stars/thu-coai/CrossWOZ?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-coai/CrossWOZ)  

- **[CUAD](https://huggingface.co/datasets/theatticusproject/cuad)** - Tasks: `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftheatticusproject%2Fcuad&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/theatticusproject/cuad) [![GitHub Stars](https://img.shields.io/github/stars/TheAtticusProject/cuad?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TheAtticusProject/cuad)  

- **[DART](https://huggingface.co/datasets/GEM/dart)** - Tasks: `Text Generation` | Mod: `Text` | Lang: `English` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGEM%2Fdart&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/GEM/dart) [![GitHub Stars](https://img.shields.io/github/stars/Yale-LILY/dart?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Yale-LILY/dart)  

- **[databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)** - Tasks: `Instruction-Following`, `Dialogue`, `Information Extraction`, `Question Answering`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdatabricks%2Fdatabricks-dolly-15k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/databricks/databricks-dolly-15k) [![GitHub Stars](https://img.shields.io/github/stars/databrickslabs/dolly?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/databrickslabs/dolly)   

- **[DialogStudio](https://huggingface.co/datasets/Salesforce/dialogstudio)** - Tasks: `Dialogue`, `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `Multi` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSalesforce%2Fdialogstudio&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Salesforce/dialogstudio) [![GitHub Stars](https://img.shields.io/github/stars/salesforce/DialogStudio?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/salesforce/DialogStudio) 

- **[Dialogue RE](https://github.com/nlpdata/dialogre)** - Tasks: `Dialogue`, `Information Extraction` | Mod: `Text` | Lang: `English` |[HomePage](https://dataset.org/dialogre/) [![GitHub Stars](https://img.shields.io/github/stars/nlpdata/dialogre?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nlpdata/dialogre) 

- **[DISC-Fin-SFT](https://huggingface.co/datasets/eggbiscuit/DISC-FIN-SFT)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Feggbiscuit%2FDISC-FIN-SFT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/eggbiscuit/DISC-FIN-SFT) [![GitHub Stars](https://img.shields.io/github/stars/FudanDISC/DISC-FinLLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FudanDISC/DISC-FinLLM)  

- **[DISC-Law-SFT](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT)** - Tasks: `Question Answering`| Mod: `Text` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FShengbinYue%2FDISC-Law-SFT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT) [![GitHub Stars](https://img.shields.io/github/stars/FudanDISC/DISC-LawLLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FudanDISC/DISC-LawLLM) 

- **[DISC-Med-SFT](https://huggingface.co/datasets/Flmc/DISC-Med-SFT)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFlmc%2FDISC-Med-SFT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Flmc/DISC-Med-SFT) [![GitHub Stars](https://img.shields.io/github/stars/FudanDISC/DISC-MedLLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FudanDISC/DISC-MedLLM) 

- **[DocRED](https://github.com/thunlp/DocRED)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/DocRED?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/DocRED) 

- **[Dolly‑15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdatabricks%2Fdatabricks-dolly-15k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/databricks/databricks-dolly-15k) 

- **[DREAM](https://github.com/nlpdata/dream)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` |[HomePage](https://dataset.org/dream/) [![GitHub Stars](https://img.shields.io/github/stars/nlpdata/dream?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nlpdata/dream) 

- **[DuoRC](https://github.com/duorc/duorc)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` |[HomePage](https://duorc.github.io/) [![GitHub Stars](https://img.shields.io/github/stars/duorc/duorc?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/duorc/duorc) 


- **[E2E](https://github.com/tuetschek/e2e-dataset)** - Tasks: `Dialogue`, `Text Generation` | Mod: `Text` | Lang: `English` |[![GitHub Stars](https://img.shields.io/github/stars/tuetschek/e2e-dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tuetschek/e2e-dataset) 

- **[EcomGPT_eval](https://github.com/Alibaba-NLP/EcomGPT)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Alibaba-NLP/EcomGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Alibaba-NLP/EcomGPT)   

- **[ELI5](https://github.com/facebookresearch/ELI5)** - Tasks: `Question Answering`, `Instruction-Following` | Mod: `Text` | Lang: `English` |[HomePage](https://facebookresearch.github.io/ELI5/explore.html) [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/ELI5?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/ELI5) 

- **[finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca)** - Tasks: `Text Classification`, `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fgbharti%2Ffinance-alpaca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/gbharti/finance-alpaca)   

- **[Firefly](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` |[![GitHub Stars](https://img.shields.io/github/stars/yangjianxin1/Firefly?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/yangjianxin1/Firefly) [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FYeungNLP%2Ffirefly-train-1.1M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) 

- **[generated_chat_0.4M](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Fgenerated_chat_0.4M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/generated_chat_0.4M) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/10M)   

- **[glaive‑function‑calling‑v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)** - Tasks: `Tool-Use` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fglaiveai%2Fglaive-function-calling-v2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)   

- **[GPTeacher](https://github.com/teknium1/GPTeacher)** - Tasks: `Instruction-Following`, `Dialogue`, `Roleplay` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/teknium1/GPTeacher?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/teknium1/GPTeacher) 

- **[GSM‑IC](https://huggingface.co/datasets/voidful/GSM-IC)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fvoidful%2FGSM-IC&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/voidful/GSM-IC) [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/GSM-IC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/GSM-IC)  

- **[HEAD-QA](https://github.com/aghie/head-qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` |[![GitHub Stars](https://img.shields.io/github/stars/aghie/head-qa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/aghie/head-qa)[HomePage](https://aghie.github.io/head-qa/)  

- **[IFLYTEK](https://huggingface.co/datasets/C-MTEB/IFlyTek-classification)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FC-MTEB%2FIFlyTek-classification&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/C-MTEB/IFlyTek-classification)   

- **[Infinity‑Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBAAI%2FInfinity-Instruct&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BAAI/Infinity-Instruct) 

- **[InstructDial](https://github.com/prakharguptaz/Instructdial)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/prakharguptaz/Instructdial?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/prakharguptaz/Instructdial) 

- **[InstructionTranslation](https://huggingface.co/datasets/theblackcat102/instruction_translations)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftheblackcat102%2Finstruction_translations&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/theblackcat102/instruction_translations)   

- **[InstructionWild](https://github.com/XueFuzhao/InstructionWild)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/XueFuzhao/InstructionWild?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/XueFuzhao/InstructionWild) 

- **[IWSLT 2017](https://huggingface.co/datasets/IWSLT/iwslt2017)** - Tasks: `Machine Translation` | Mod: `Text` | Lang: `English`, `Dutch`, `German`, `Spanish`, `Romanian`, `Italian`| [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fiwslt2017&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/IWSLT/iwslt2017) [![GitHub Stars](https://img.shields.io/github/stars/puttisandev/iwslt2017?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/puttisandev/iwslt2017)  

- **[Japanese Alpaca](https://huggingface.co/datasets/fujiki/japanese_alpaca_data)** - Tasks: `Instruction-Following`, `Dialogue`, `Text Classification`, `Machine Translation` | Mod: `Text` | Lang: `Japanese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ffujiki%2Fjapanese_alpaca_data&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/fujiki/japanese_alpaca_data) [![GitHub Stars](https://img.shields.io/github/stars/Tatsu-lab/stanford_alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Tatsu-lab/stanford_alpaca) 

- **[kollm-converations](https://huggingface.co/datasets/davidkim205/kollm-converations)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `Korean` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fdavidkim205%2Fkollm-converations&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/davidkim205/kollm-converations)   

- **[LaMini-instruction](https://huggingface.co/datasets/MBZUAI/LaMini-instruction)** - Tasks: `Instruction-Following`, `Summarization` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMBZUAI%2FLaMini-instruction&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MBZUAI/LaMini-instruction) [![GitHub Stars](https://img.shields.io/github/stars/mbzuai-nlp/LaMini-LM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mbzuai-nlp/LaMini-LM)   

- **[LawGPT_zh](https://github.com/LiuHC0428/LAW-GPT)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/LiuHC0428/LAW-GPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LiuHC0428/LAW-GPT)   

- **[Lawyer LLaMA_sft](https://github.com/AndrewZhe/lawyer-llama)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/AndrewZhe/lawyer-llama?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AndrewZhe/lawyer-llama) 

- **[LCSTS](https://huggingface.co/datasets/hugcyp/LCSTS)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhugcyp%2FLCSTS&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hugcyp/LCSTS)   

- **[lima](https://huggingface.co/datasets/GAIR/lima)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGAIR%2Flima&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/GAIR/lima)   

- **[lithuanian-qa-v1](https://huggingface.co/datasets/neurotechnology/lithuanian-qa-v1)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Lt` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fneurotechnology%2Flithuanian-qa-v1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/neurotechnology/lithuanian-qa-v1)   

- **[lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Flmsys-chat-1m&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)   

- **[LongForm](https://huggingface.co/datasets/akoksal/LongForm)** - Tasks: `Instruction-Following`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fakoksal%2FLongForm&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/akoksal/LongForm) [![GitHub Stars](https://img.shields.io/github/stars/akoksal/LongForm?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/akoksal/LongForm)   

- **[LongWriter-6k](https://huggingface.co/datasets/zai-org/LongWriter-6k)** - Tasks: `Instruction-Following`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fzai-org%2FLongWriter-6k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/zai-org/LongWriter-6k) [![GitHub Stars](https://img.shields.io/github/stars/THUDM/LongWriter?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/LongWriter)   

- **[Luotuo-QA-B](https://huggingface.co/datasets/Logic123456789/Luotuo-QA-B)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese`, `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FLogic123456789%2FLuotuo-QA-B&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Logic123456789/Luotuo-QA-B) [![GitHub Stars](https://img.shields.io/github/stars/LC1332/Luotuo-QA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LC1332/Luotuo-QA) 

- **[MARC](https://huggingface.co/datasets/mteb/amazon_reviews_multi)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `German` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmteb%2Famazon_reviews_multi&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mteb/amazon_reviews_multi)  

- **[math](https://huggingface.co/datasets/EleutherAI/hendrycks_math)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FEleutherAI%2Fhendrycks_math&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/EleutherAI/hendrycks_math) [![GitHub Stars](https://img.shields.io/github/stars/hendrycks/math?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hendrycks/math)   

- **[Math23K](https://huggingface.co/datasets/SUSTech/math23k)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/SCNU203/Math23k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SCNU203/Math23k) [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSUSTech%2Fmath23k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/SUSTech/math23k)  

- **[MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTIGER-Lab%2FMathInstruct&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) [![GitHub Stars](https://img.shields.io/github/stars/TIGER-AI-Lab/MAmmoTH?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TIGER-AI-Lab/MAmmoTH)   

- **[MathQA](https://huggingface.co/datasets/allenai/math_qa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fmath_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/math_qaa)  

- **[MediaSum](https://huggingface.co/datasets/ccdv/mediasum)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fccdv%2Fmediasum&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ccdv/mediasum) [![GitHub Stars](https://img.shields.io/github/stars/zcgzcgzcg1/MediaSum?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/zcgzcgzcg1/MediaSum/) 

- **[medical](https://huggingface.co/datasets/shibing624/medical)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fshibing624%2Fmedical&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shibing624/medical) [![GitHub Stars](https://img.shields.io/github/stars/shibing624/MedicalGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/shibing624/MedicalGPT)   

- **[MedNLI](https://huggingface.co/datasets/bigbio/mednli)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigbio%2Fmednli&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigbio/mednli) [![GitHub Stars](https://img.shields.io/github/stars/jgc128/mednli?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jgc128/mednli)   

- **[MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmeta-math%2FMetaMathQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/meta-math/MetaMathQA) [![GitHub Stars](https://img.shields.io/github/stars/meta-math/MetaMath?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/meta-math/MetaMath)   

- **[METS-CoV](https://github.com/YLab-Open/METS-CoV)** - Tasks: `Text Classification`, `Sentiment Analysis` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/YLab-Open/METS-CoV?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/YLab-Open/METS-CoV) 

- **[MOSS SFT data](https://huggingface.co/datasets/OpenMOSS-Team/moss-003-sft-data)** - Tasks: `Dialogue`, `Text Classification` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/OpenLMLab/MOSS?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenLMLab/MOSS/tree/main/SFT_data) [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenMOSS-Team%2Fmoss-003-sft-data&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenMOSS-Team/moss-003-sft-data)  

- **[MRPC](https://huggingface.co/datasets/SetFit/mrpc)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSetFit%2Fmrpc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/SetFit/mrpc)   

- **[MS MARCO](https://github.com/microsoft/MSMARCO-Question-Answering)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/microsoft/MSMARCO-Question-Answering?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/microsoft/MSMARCO-Question-Answering) 

- **[MSRA_NER](https://huggingface.co/datasets/levow/msra_ner)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flevow%2Fmsra_ner&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/levow/msra_ner)   

- **[MultiNews](https://huggingface.co/datasets/alexfabbri/multi_news)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Falexfabbri%2Fmulti_news&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/alexfabbri/multi_news)[![GitHub Stars](https://img.shields.io/github/stars/Alex-Fabbri/Multi-News?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Alex-Fabbri/Multi-News) 

- **[MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnyu-mll%2Fmulti_nli&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nyu-mll/multi_nli)   

- **[MultiRC](https://huggingface.co/datasets/CogComp/eraser_multi_rc)** - Tasks: `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FCogComp%2Feraser_multi_rc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/CogComp/eraser_multi_rc) [![GitHub Stars](https://img.shields.io/github/stars/CogComp/multirc?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CogComp/multirc)   

- **[multiturn_chat_0.8M](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Fmultiturn_chat_0.8M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/10M)   

- **[MultiWOZ](https://github.com/budzianowski/multiwoz)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/budzianowski/multiwoz?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/budzianowski/multiwoz) 

- **[Natural Questions](https://github.com/google-research-datasets/natural-questions)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/natural-questions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/natural-questions) 

- **[natural-instructions](https://huggingface.co/datasets/Muennighoff/natural-instructions)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMuennighoff%2Fnatural-instructions&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Muennighoff/natural-instructions) [![GitHub Stars](https://img.shields.io/github/stars/allenai/natural-instructions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/natural-instructions)   

- **[NaturalReasoning](https://huggingface.co/datasets/facebook/natural_reasoning)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ffacebook%2Fnatural_reasoning&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/facebook/natural_reasoning)   

- **[Newsroom](https://huggingface.co/datasets/lil-lab/newsroom)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flil-lab%2Fnewsroom&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lil-lab/newsroom)   

- **[no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceH4%2Fno_robots&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceH4/no_robots)   

- **[OCNLI](https://github.com/cluebenchmark/OCNLI)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/cluebenchmark/OCNLI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/cluebenchmark/OCNLI)   

- **[OIG](https://huggingface.co/datasets/laion/OIG)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text`, `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flaion%2FOIG&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/laion/OIG) 

- **[OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fopenbookqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/openbookqa) [![GitHub Stars](https://img.shields.io/github/stars/allenai/OpenBookQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/OpenBookQA)   

- **[OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1)** - Tasks: `Instruction-Following`, `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnvidia%2FOpenMathInstruct-1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) [![GitHub Stars](https://img.shields.io/github/stars/Kipok/NeMo-Skills?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Kipok/NeMo-Skills) 

- **[OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpen-Orca%2FOpenOrca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Open-Orca/OpenOrca) 

- **[OpenR1‑Math‑220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopen-r1%2FOpenR1-Math-220k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)   

- **[Open‑PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend)** - Tasks: `Dialogue`, `Instruction-Following`| Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmlabonne%2Fopen-perfectblend&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mlabonne/open-perfectblend) 

- **[orca-math-word-problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmicrosoft%2Forca-math-word-problems-200k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)   

- **[orca‑chat](https://huggingface.co/datasets/Open-Orca/OpenOrca)** - Tasks: `Dialogue`, `Instruction-Following`, `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpen-Orca%2FOpenOrca&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Open-Orca/OpenOrca)   

- **[PIQA](https://huggingface.co/datasets/ybisk/piqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fybisk%2Fpiqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/piqa)   

- **[PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/michael-wzhu/PromptCBLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/michael-wzhu/PromptCBLUE)   

- **[PromptSource](https://github.com/bigscience-workshop/promptsource)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/bigscience-workshop/promptsource?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/bigscience-workshop/promptsource) 

- **[PsyQA](https://huggingface.co/datasets/lsy641/PsyQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flsy641%2FPsyQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lsy641/PsyQA) [![GitHub Stars](https://img.shields.io/github/stars/thu-coai/PsyQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-coai/PsyQA) 

- **[PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fqiaojin%2Fpubmed_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/qiaojin/PubMedQA) [![GitHub Stars](https://img.shields.io/github/stars/pubmedqa/pubmedqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/pubmedqa/pubmedqa)   

- **[QASC](https://huggingface.co/datasets/allenai/qasc)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fqasc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/qasc)   

- **[QED](https://github.com/google-research-datasets/QED)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/QED?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/QED)   

- **[QQP](https://huggingface.co/datasets/merve/qqp)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmerve%2Fqqp&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/merve/qqp)   

- **[QuaRTz](https://huggingface.co/datasets/allenai/quartz)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fquartz&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/quartz)   

- **[Quoref](https://huggingface.co/datasets/allenai/quoref)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fquoref&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/quoref)   

- **[RACE](https://huggingface.co/datasets/ehovy/race)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fehovy%2Frace&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ehovy/race)   

- **[RefGPT](https://huggingface.co/datasets/Mutonix/RefGPT-Code-ds)** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMutonix%2FRefGPT-Code-ds&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Mutonix/RefGPT-Code-ds) [![GitHub Stars](https://img.shields.io/github/stars/mutonix/RefGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mutonix/RefGPT)  

- **[Resume](https://github.com/jiesutd/LatticeLSTM)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/jiesutd/LatticeLSTM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jiesutd/LatticeLSTM) 

- **[ROPES](https://huggingface.co/datasets/allenai/ropes)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fropes&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/ropes)   

- **[RTE](https://huggingface.co/datasets/nyu-mll/glue/viewer/rte/train)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnyu-mll%2Fglue&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nyu-mll/glue)   

- **[Safety Prompts](https://huggingface.co/datasets/thu-coai/Safety-Prompts)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fthu-coai%2FSafety-Prompts&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/thu-coai/Safety-Prompts) [![GitHub Stars](https://img.shields.io/github/stars/thu-coai/Safety-Prompts?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-coai/Safety-Prompts)   

- **[SAMSum](https://huggingface.co/datasets/knkarthick/samsum)** - Tasks: `Summarization`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fknkarthick%2Fsamsum&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/knkarthick/samsum)   

- **[school_math_0.25M](https://huggingface.co/datasets/BelleGroup/school_math_0.25M)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Fschool_math_0.25M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/school_math_0.25M) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/10M)   

- **[SciQ](https://huggingface.co/datasets/allenai/sciq)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fsciq&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/sciq)   

- **[self-instruct](https://github.com/yizhongw/self-instruct)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/yizhongw/self-instruct?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/yizhongw/self-instruct)   

- **[Sentiment140](https://huggingface.co/datasets/stanfordnlp/sentiment140)** - Tasks: `Sentiment Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstanfordnlp%2Fsentiment140&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stanfordnlp/sentiment140)   

- **[sft_en](https://huggingface.co/datasets/TigerResearch/sft_en)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Fsft_en&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/sft_en) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)   

- **[sft_zh](https://huggingface.co/datasets/TigerResearch/sft_zh)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Fsft_zh&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/sft_zh) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)   

- **[ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k)** - Tasks: `Machine Translation`, `Question Answering` | Mod: `Text` | Lang: `English`, `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FshareAI%2FShareGPT-Chinese-English-90k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) [![GitHub Stars](https://img.shields.io/github/stars/CrazyBoyM/llama2-Chinese-chat?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CrazyBoyM/llama2-Chinese-chat)   

- **[ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fanon8231489123%2FShareGPT_Vicuna_unfiltered&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)   

- **[smolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceTB%2Fsmoltalk&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) [![GitHub Stars](https://img.shields.io/github/stars/huggingface/smollm?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/huggingface/smollm/tree/main/text/data/smoltalk)   

- **[SNLI](https://huggingface.co/datasets/stanfordnlp/snli)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstanfordnlp%2Fsnli&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stanfordnlp/snli)   

- **[Spider](https://huggingface.co/datasets/xlangai/spider)** - Tasks: `Code Generation`, `Question Answering` | Mod: `Text`, `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fxlangai%2Fspider&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/xlangai/spider) [![GitHub Stars](https://img.shields.io/github/stars/taoyds/spider?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/taoyds/spider)   

- **[SQuAD](https://huggingface.co/datasets/rajpurkar/squad)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Frajpurkar%2Fsquad&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/rajpurkar/squad)   

- **[SST-2](https://huggingface.co/datasets/stanfordnlp/sst2)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstanfordnlp%2Fsst2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stanfordnlp/sst2)   

- **[stack-exchange-paired](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flvwerra%2Fstack-exchange-paired&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)   

- **[StackOverflow post](https://huggingface.co/datasets/mikex86/stackoverflow-posts)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmikex86%2Fstackoverflow-posts&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mikex86/stackoverflow-posts) [![GitHub Stars](https://img.shields.io/github/stars/StackExchange/StackExchange.DataExplorer?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/StackExchange/StackExchange.DataExplorer)   

- **[STSB](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt)** - Tasks: `Semantic Similarity` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhilipMay%2Fstsb_multi_mt&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/PhilipMay/stsb_multi_mt) [![GitHub Stars](https://img.shields.io/github/stars/PhilipMay/stsb-multi-mt?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/PhilipMay/stsb-multi-mt)   

- **[SUPER-NATURAL INSTRUCTIONS](https://github.com/allenai/natural-instructions)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/allenai/natural-instructions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/natural-instructions) 

- **[SVAMP](https://github.com/arkilpatel/SVAMP)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/arkilpatel/SVAMP?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/arkilpatel/SVAMP)   

- **[TACRED](https://huggingface.co/datasets/DFKI-SLT/tacred)** - Tasks: `Relation Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FDFKI-SLT%2Ftacred&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/DFKI-SLT/tacred)   

- **[Taobao NER](https://github.com/allanj/ner_incomplete_annotation)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/allanj/ner_incomplete_annotation?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allanj/ner_incomplete_annotation)   

- **[TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTIGER-Lab%2FTheoremQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) [![GitHub Stars](https://img.shields.io/github/stars/wenhuchen/TheoremQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/wenhuchen/TheoremQA/tree/main)   

- **[THUCNews](https://github.com/thunlp/THUCTC)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/THUCTC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/THUCTC)   

- **[tigerbot-law-plugin](https://huggingface.co/datasets/TigerResearch/tigerbot-law-plugin)** - Tasks: `Legal`, `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTigerResearch%2Ftigerbot-law-plugin&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TigerResearch/tigerbot-law-plugin) [![GitHub Stars](https://img.shields.io/github/stars/TigerResearch/TigerBot?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TigerResearch/TigerBot)   

- **[TNEWS](https://huggingface.co/datasets/mteb/TNews)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmteb%2FTNews&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mteb/TNews)   

- **[Traditional Chinese Alpaca](https://github.com/ntunlplab/traditional-chinese-alpaca)** - Tasks: `Instruction-Following`, `Dialogue`, `Machine Translation` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/ntunlplab/traditional-chinese-alpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ntunlplab/traditional-chinese-alpaca)   

- **[train_0.5M_CN](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Ftrain_0.5M_CN&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)   

- **[train_1M_CN](https://huggingface.co/datasets/BelleGroup/train_1M_CN)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBelleGroup%2Ftrain_1M_CN&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/BelleGroup/train_1M_CN) [![GitHub Stars](https://img.shields.io/github/stars/LianjiaTech/BELLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M)   

- **[TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmandarjoshi%2Ftrivia_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mandarjoshi/trivia_qa) [![GitHub Stars](https://img.shields.io/github/stars/mandarjoshi90/triviaqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mandarjoshi90/triviaqa)   

- **[TSI-v0](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0)** - Tasks: `Text Classification`, `TokenClassification`, `MultipleChoice` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftasksource%2Ftasksource-instruct-v0&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tasksource/tasksource-instruct-v0) [![GitHub Stars](https://img.shields.io/github/stars/sileod/tasksource?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sileod/tasksource)   

- **[UltraChat_200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceH4%2Fultrachat_200k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) [![GitHub Stars](https://img.shields.io/github/stars/thunlp/UltraChat?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/UltraChat)   

- **[Unnatural Instructions](https://huggingface.co/datasets/mrm8488/unnatural-instructions-full)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fnatural-instructions&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mrm8488/unnatural-instructions-full) [![GitHub Stars](https://img.shields.io/github/stars/orhonovich/unnatural-instructions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/orhonovich/unnatural-instructions)   

- **[Vicuna Dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fanon8231489123%2FShareGPT_Vicuna_unfiltered&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) [![GitHub Stars](https://img.shields.io/github/stars/lm-sys/FastChat?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lm-sys/FastChat)   

- **[WANLI](https://huggingface.co/datasets/alisawuffles/WANLI)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Falisawuffles%2FWANLI&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/alisawuffles/WANLI)   

- **[webglm-qa](https://huggingface.co/datasets/THUDM/webglm-qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTHUDM%2Fwebglm-qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/THUDM/webglm-qa) [![GitHub Stars](https://img.shields.io/github/stars/THUDM/WebGLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/WebGLM)   

- **[webgpt_comparisons](https://huggingface.co/datasets/openai/webgpt_comparisons)** - Tasks: `Instruction-Following`, `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2Fwebgpt_comparisons&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/webgpt_comparisons)   

- **[WebMedQA](https://huggingface.co/datasets/zirui3/webMedQA-instructions)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fzirui3%2FwebMedQA-instructions&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/zirui3/webMedQA-instructions) [![GitHub Stars](https://img.shields.io/github/stars/hejunqing/webMedQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hejunqing/webMedQA)   

- **[WebNLG](https://huggingface.co/datasets/GEM/web_nlg)** - Tasks: `Text Generation` | Mod: `Text` | Lang: `English` ,`Russian`| [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGEM%2Fweb_nlg&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/GEM/web_nlg) 

- **[Weibo NER](https://github.com/hltcoe/golden-horse)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/hltcoe/golden-horse?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hltcoe/golden-horse)   

- **[WikiHow](https://github.com/HiDhineshRaja/WikiHow-Dataset)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/HiDhineshRaja/WikiHow-Dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HiDhineshRaja/WikiHow-Dataset)   

- **[WildChat](https://huggingface.co/datasets/allenai/WildChat)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2FWildChat&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/WildChat)   

- **[WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fwildguardmix&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/wildguardmix)   

- **[WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fwildjailbreak&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/wildjailbreak)   

- **[Wizard-LM-Chinese-instruct-evol](https://huggingface.co/datasets/silk-road/Wizard-LM-Chinese-instruct-evol)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsilk-road%2FWizard-LM-Chinese-instruct-evol&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/silk-road/Wizard-LM-Chinese-instruct-evol) [![GitHub Stars](https://img.shields.io/github/stars/LC1332/Chinese-alpaca-lora?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LC1332/Chinese-alpaca-lora)   

- **[WizardLM_evol_instruct_V2_196k](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FWizardLM%2FWizardLM_evol_instruct_V2_196k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V2_196k) [![GitHub Stars](https://img.shields.io/github/stars/nlpxucan/WizardLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nlpxucan/WizardLM)   

- **[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcsebuetnlp%2Fxlsum&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/csebuetnlp/xlsum)   

- **[xP3](https://huggingface.co/datasets/bigscience/xP3)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigscience%2FxP3&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigscience/xP3)   

- **[Youku NER](https://github.com/allanj/ner_incomplete_annotation)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/allanj/ner_incomplete_annotation?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allanj/ner_incomplete_annotation) 



<a id="text-alignment-rlhf"></a>
#### Alignment / RL
- **[Anthropic HH‑RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FAnthropic%2Fhh-rlhf&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Anthropic/hh-rlhf)  

- **[chatbot_arena conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)** - Tasks: `Dialogue` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Fchatbot_arena_conversations&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)  

- **[FineGrainedRLHF](https://github.com/allenai/FineGrainedRLHF)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/allenai/FineGrainedRLHF?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/FineGrainedRLHF)  

- **[GPT-4-LLM Dataset](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Instruction-Tuning-with-GPT-4/GPT-4-LLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)  

- **[HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)** - Tasks: `Instruction-Following`, `Dialogue`, `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHello-SimpleAI%2FHC3&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Hello-SimpleAI/HC3)  

- **[HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer)** - Tasks: `Instruction-Following`, `Dialogue`，`Summarization`, `Question Answering`， `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnvidia%2FHelpSteer&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nvidia/HelpSteer)  

- **[HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnvidia%2FHelpSteer2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nvidia/HelpSteer2)  

- **[OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenAssistant%2Foasst1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenAssistant/oasst1) [![GitHub Stars](https://img.shields.io/github/stars/LAION-AI/Open-Assistant?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LAION-AI/Open-Assistant)

- **[OASST2 (final)](https://huggingface.co/datasets/OpenAssistant/oasst2)** - Tasks: `Dialogue`, `Instruction-Following` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenAssistant%2Foasst2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenAssistant/oasst2) [![GitHub Stars](https://img.shields.io/github/stars/LAION-AI/Open-Assistant?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LAION-AI/Open-Assistant)

- **[OpenAI Summarization Comparison](https://huggingface.co/datasets/openai/summarize_from_feedback)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2Fsummarize_from_feedback&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/summarize_from_feedback)  

- **[OpenAI WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2Fwebgpt_comparisons&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/webgpt_comparisons)  

- **[PKU‑SafeRLHF‑10K](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPKU-Alignment%2FPKU-SafeRLHF-10K&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-10K)  

- **[PRM800K](https://github.com/openai/prm800k)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/openai/prm800k?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/prm800k)  

- **[SHP](https://huggingface.co/datasets/stanfordnlp/SHP)** - Tasks: `Dialogue`, `Human Preference` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fstanfordnlp%2FSHP&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stanfordnlp/SHP)  

- **[TabMWP](https://github.com/lupantech/PromptPG)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text`, `Table` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/lupantech/PromptPG?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lupantech/PromptPG)

- **[UltraFeedback (cleaned binarized)](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fargilla%2Fultrafeedback-binarized-preferences-cleaned&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)  



<a id="text-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- **[aclue](https://huggingface.co/datasets/tyouisen/aclue)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftyouisen%2Faclue&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tyouisen/aclue) [![GitHub Stars](https://img.shields.io/github/stars/isen-zhang/aclue?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/isen-zhang/aclue)   

- **[AlignBench](https://github.com/THUDM/AlignBench)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/THUDM/AlignBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/AlignBench)   

- **[ArabicMMLU](https://huggingface.co/datasets/MBZUAI/ArabicMMLU)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Arabic` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMBZUAI%2FArabicMMLU&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MBZUAI/ArabicMMLU) [![GitHub Stars](https://img.shields.io/github/stars/mbzuai-nlp/ArabicMMLU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mbzuai-nlp/ArabicMMLU)   

- **[ArabLegalEval 2024-8](https://github.com/Thiqah/ArabLegalEval)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Arabic` | [![GitHub Stars](https://img.shields.io/github/stars/Thiqah/ArabLegalEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Thiqah/ArabLegalEval)   

- **[ARB](https://github.com/TheDuckAI/arb)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/TheDuckAI/arb?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TheDuckAI/arb)   

- **[ARC](https://huggingface.co/datasets/allenai/ai2_arc)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fai2_arc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/ai2_arc)   

- **[ARES](https://github.com/stanford-futuredata/ARES)** - Tasks: `Evaluation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/stanford-futuredata/ARES?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/stanford-futuredata/ARES)   

- **[BBF-CFLEB](https://github.com/ssymmetry/BBT-FinCUGE-Applications)** - Tasks: `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/ssymmetry/BBT-FinCUGE-Applications?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ssymmetry/BBT-FinCUGE-Applications)   

- **[BBH](https://github.com/suzgunmirac/BIG-Bench-Hard)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/suzgunmirac/BIG-Bench-Hard?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/suzgunmirac/BIG-Bench-Hard)   

- **[BIG-Bench](https://github.com/google/BIG-bench)** - Tasks: `Reasoning`, `Common Sense Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google/BIG-bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google/BIG-bench)   

- **[BoolQ](https://github.com/google-research-datasets/boolean-questions)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/boolean-questions?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/boolean-questions)   

- **[BOSS](https://github.com/lifan-yuan/OOD_NLP)** - Tasks: `Sentiment Analysis`, `Toxicity Detection`, `Natural Language Inference`, `Named Entity Recognition`, `Extractive Question Answering` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/lifan-yuan/OOD_NLP?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lifan-yuan/OOD_NLP)   

- **[BUSTM](https://github.com/xiaobu-coai/BUSTM)** - Tasks: `Semantic Matching` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/xiaobu-coai/BUSTM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/xiaobu-coai/BUSTM)   

- **[C-CLUE](https://github.com/jizijing/C-CLUE)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/jizijing/C-CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jizijing/C-CLUE)   

- **[C3 Bench 2024-5](https://huggingface.co/datasets/tencent/C3-BenchMark)** - Tasks: `Text Classification`, `Information Extraction`, `Machine Translation`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftencent%2FC3-BenchMark&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tencent/C3-BenchMark)    

- **[CBLUE](https://github.com/CBLUEbenchmark/CBLUE)** - Tasks: `Information Extraction`, `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CBLUEbenchmark/CBLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CBLUEbenchmark/CBLUE)   

- **[ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fceval%2Fceval-exam&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ceval/ceval-exam) [![GitHub Stars](https://img.shields.io/github/stars/SJTU-LIT/ceval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SJTU-LIT/ceval)   

- **[CG-Eval](https://huggingface.co/datasets/Besteasy/CG-Eval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBesteasy%2FCG-Eval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Besteasy/CG-Eval) [![GitHub Stars](https://img.shields.io/github/stars/Felixgithub2017/CG-Eval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Felixgithub2017/CG-Eval)   

- **[Chinese-SimpleQA](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenStellarTeam%2FChinese-SimpleQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SimpleQA) [![GitHub Stars](https://img.shields.io/github/stars/OpenStellarTeam/ChineseSimpleQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenStellarTeam/ChineseSimpleQA)    

- **[ChineseFactEval](https://github.com/GAIR-NLP/factool)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/GAIR-NLP/factool?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/GAIR-NLP/factool)   

- **[Choice-75](https://github.com/JoeyHou/branching)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/JoeyHou/branching?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/JoeyHou/branching)   

- **[CLEVA](https://github.com/LaVi-Lab/CLEVA)** - Tasks: `Evaluation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/LaVi-Lab/CLEVA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/LaVi-Lab/CLEVA)   

- **[CLongEval](https://huggingface.co/datasets/zexuanqiu22/CLongEval)** - Tasks: `Question Answering`, `Summarization`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fzexuanqiu22%2FCLongEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/zexuanqiu22/CLongEval) [![GitHub Stars](https://img.shields.io/github/stars/zexuanqiu/CLongEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/zexuanqiu/CLongEval)   

- **[CLUE](https://github.com/CLUEbenchmark/CLUE)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUE)   

- **[CLUEWSC2020](https://github.com/CLUEbenchmark/CLUEWSC2020)** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUEWSC2020?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUEWSC2020)    

- **[cmath](https://huggingface.co/datasets/weitianwen/cmath)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fweitianwen%2Fcmath&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/weitianwen/cmath) [![GitHub Stars](https://img.shields.io/github/stars/XiaoMi/cmath?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/XiaoMi/cmath)   

- **[CMB](https://huggingface.co/datasets/FreedomIntelligence/CMB)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFreedomIntelligence%2FCMB&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/FreedomIntelligence/CMB) [![GitHub Stars](https://img.shields.io/github/stars/FreedomIntelligence/CMB?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FreedomIntelligence/CMB)   

- **[cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhaonan-li%2Fcmmlu&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/haonan-li/cmmlu) [![GitHub Stars](https://img.shields.io/github/stars/haonan-li/CMMLU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/haonan-li/CMMLU)   

- **[CMNLI](https://github.com/CLUEbenchmark/CLUE)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/CLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/CLUE)   

- **[CMRC2019](https://github.com/ymcui/cmrc2019)** - Tasks: `Cloze Test` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/ymcui/cmrc2019?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ymcui/cmrc2019)   

- **[CN-SarcasmBench](https://huggingface.co/datasets/Devon018/CN-SarcasmBench)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FDevon018%2FCN-SarcasmBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Devon018/CN-SarcasmBench)  

- **[CoLA](https://github.com/nyu-mll/CoLA)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nyu-mll/CoLA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nyu-mll/CoLA)    

- **[CommitmentBank](https://github.com/mcdm/CommitmentBank)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/mcdm/CommitmentBank?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mcdm/CommitmentBank)   

- **[CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftau%2Fcommonsense_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tau/commonsense_qa) [![GitHub Stars](https://img.shields.io/github/stars/jonathanherzig/commonsenseqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/jonathanherzig/commonsenseqa)    

- **[CoNLL2003](https://huggingface.co/datasets/eriktks/conll2003)** - Tasks: `Text classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fconll2003&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/eriktks/conll2003)   

- **[CosmosQA](https://huggingface.co/datasets/allenai/cosmos_qa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcosmos_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/cosmos_qa) [![GitHub Stars](https://img.shields.io/github/stars/wilburOne/cosmosqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/wilburOne/cosmosqa/)    

- **[Counting-Stars](https://github.com/nick7nlp/counting-stars)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nick7nlp/counting-stars?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nick7nlp/counting-stars)   

- **[CRAG](https://github.com/facebookresearch/CRAG)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/CRAG?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/CRAG)    

- **[CREAK](https://github.com/yasumasaonoe/creak)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/yasumasaonoe/creak?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/yasumasaonoe/creak)   

- **[CrowS-Pairs](https://github.com/nyu-mll/crows-pairs)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nyu-mll/crows-pairs?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nyu-mll/crows-pairs)   

- **[CSCD-IME](https://github.com/nghuyong/cscd-ime)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nghuyong/cscd-ime?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nghuyong/cscd-ime)   

- **[CUGE](https://github.com/TsinghuaAI/CUGE)** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/TsinghuaAI/CUGE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TsinghuaAI/CUGE)    

- **[DebateQA 2024-8](https://github.com/pillowsofwind/DebateQA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/pillowsofwind/DebateQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/pillowsofwind/DebateQA)   

- **[decaNLP](https://github.com/salesforce/decaNLP)** - Tasks: `Question Answering`, `Machine Translation`, `Summarization`, `Natural Language Inference` | Mod: `Text` | Lang: `English`, `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/salesforce/decaNLP?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/salesforce/decaNLP)   

- **[DROP](https://huggingface.co/datasets/ucinlp/drop)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fucinlp%2Fdrop&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ucinlp/drop)    

- **[DuQM](https://github.com/baidu/DuReader/tree/master/DuQM)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/baidu/DuReader?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/baidu/DuReader)   

- **[DuReader Checklist](https://github.com/baidu/DuReader/tree/master/DuReader-Checklist)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/baidu/DuReader?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/baidu/DuReader)   

- **[DuReader Robust](https://github.com/baidu/DuReader/tree/master/DuReader-Robust)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/baidu/DuReader?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/baidu/DuReader)   

- **[EmotionBench](https://github.com/CUHK-ARISE/EmotionBench)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/CUHK-ARISE/EmotionBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CUHK-ARISE/EmotionBench)   

- **[EPRSTMT](https://github.com/CLUEbenchmark/FewCLUE)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/FewCLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/FewCLUE)

- **[FACTOR](https://github.com/AI21Labs/factor)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/AI21Labs/factor?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AI21Labs/factor)   

- **[FActScore](https://github.com/shmsw25/FActScore)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/shmsw25/FActScore?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/shmsw25/FActScore)   

- **[FactualityPrompt](https://github.com/nayeon7lee/FactualityPrompt)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nayeon7lee/FactualityPrompt?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nayeon7lee/FactualityPrompt)   

- **[FairEval](https://github.com/i-Eval/FairEval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/i-Eval/FairEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/i-Eval/FairEval)   

- **[Few-NERD](https://github.com/thunlp/Few-NERD)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/Few-NERD?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/Few-NERD)   

- **[FewCLUE](https://github.com/CLUEbenchmark/FewCLUE)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/FewCLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/FewCLUE)   

- **[FewRel](https://github.com/thunlp/fewrel)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/fewrel?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/fewrel)   

- **[FinancelQ](https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Duxiaoman-DI/XuanYuan?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ)   

- **[FinBen](https://github.com/The-FinAI/FinBen)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/The-FinAI/FinBen?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/The-FinAI/FinBen)   

- **[FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceTB%2Ffinemath&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceTB/finemath)   

- **[FinEval](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSUFE-AIFLM-Lab%2FFinEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/SUFE-AIFLM-Lab/FinEval) [![GitHub Stars](https://img.shields.io/github/stars/SUFE-AIFLM-Lab/FinEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SUFE-AIFLM-Lab/FinEval)   

- **[FlagEval](https://github.com/FlagOpen/FlagEval)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/FlagOpen/FlagEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FlagOpen/FlagEval)   

- **[FLUE](https://huggingface.co/datasets/GETALP/flue)** - Tasks: `Sentiment Analysis`, `Text Classification`, `Named Entity Recognition`, `Question Answering` | Mod: `Text` | Lang: `French` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FGETALP%2Fflue&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/GETALP/flue)   

- **[FreshQA](https://github.com/freshllms/freshqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/freshllms/freshqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/freshllms/freshqa)   

- **[GeoBench](https://github.com/davendw49/k2)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/davendw49/k2?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/davendw49/k2)   

- **[GLUE](https://github.com/nyu-mll/GLUE-baselines)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nyu-mll/GLUE-baselines?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nyu-mll/GLUE-baselines)   

- **[GLUE-X](https://github.com/YangLinyi/GLUE-X)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/YangLinyi/GLUE-X?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/YangLinyi/GLUE-X)   

- **[GPQA](https://huggingface.co/datasets/Idavidrein/gpqa)** - Tasks: `Question Answering`， `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FIdavidrein%2Fgpqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Idavidrein/gpqa) [![GitHub Stars](https://img.shields.io/github/stars/idavidrein/gpqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/idavidrein/gpqa)   

- **[GSM8K](https://huggingface.co/datasets/openai/gsm8k)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2Fgsm8k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/gsm8k) [![GitHub Stars](https://img.shields.io/github/stars/openai/grade-school-math?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/grade-school-math)   

- **[HalluDial 2024-6](https://github.com/FlagOpen/HalluDial)** - Tasks: `Dialogue`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/FlagOpen/HalluDial?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FlagOpen/HalluDial)   

- **[HalluQA](https://github.com/xiami2019/HalluQA/)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/xiami2019/HalluQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/xiami2019/HalluQA/)   

- **[HaluEval](https://github.com/RUCAIBox/HaluEval)** - Tasks: `Question Answering`, `Dialogue` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/RUCAIBox/HaluEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/RUCAIBox/HaluEval)   

- **[healthsearchqa](https://huggingface.co/datasets/katielink/healthsearchqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fkatielink%2Fhealthsearchqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/katielink/healthsearchqa)   

- **[HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FRowan%2Fhellaswag&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Rowan/hellaswag) [![GitHub Stars](https://img.shields.io/github/stars/rowanz/hellaswag?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/rowanz/hellaswag)   

- **[HOTPOTQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhotpot_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hotpotqa/hotpot_qa)   

- **[huatuo26M-testdatasets](https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFreedomIntelligence%2Fhuatuo26M-testdatasets&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/FreedomIntelligence/huatuo26M-testdatasets) [![GitHub Stars](https://img.shields.io/github/stars/FreedomIntelligence/Huatuo-26M?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/FreedomIntelligence/Huatuo-26M)   

- **[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44)** - Tasks: `Machine Translation`, `Text Classification`, `Information Extraction` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FCohereLabs%2Finclude-base-44&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/CohereLabs/include-base-44) 

- **[InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench)** - Tasks: `Question Answering`, `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fxinrongzhang2022%2FInfiniteBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench) [![GitHub Stars](https://img.shields.io/github/stars/OpenBMB/InfiniteBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenBMB/InfiniteBench)   

- **[JEC-QA](https://github.com/thunlp/jec-qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/thunlp/jec-qa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thunlp/jec-qa)   

- **[kobest_v1](https://huggingface.co/datasets/skt/kobest_v1)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Korean` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fskt%2Fkobest_v1&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/skt/kobest_v1)   

- **[KoLA](https://github.com/THU-KEG/KoLA)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/THU-KEG/KoLA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THU-KEG/KoLA)   

- **[LAiW](https://github.com/Dai-shen/LAiW)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Dai-shen/LAiW?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Dai-shen/LAiW)   

- **[LAMBADA](https://huggingface.co/datasets/cimec/lambada)** - Tasks: `Word Prediction`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flambada&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cimec/lambada)   

- **[LawBench](https://github.com/open-compass/LawBench)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/open-compass/LawBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/open-compass/LawBench)   

- **[LCQMC](https://huggingface.co/datasets/mteb/LCQMC)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmteb%2FLCQMC&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mteb/LCQMC)   

- **[legalbench](https://huggingface.co/datasets/nguha/legalbench)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fnguha%2Flegalbench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/nguha/legalbench) [![GitHub Stars](https://img.shields.io/github/stars/HazyResearch/legalbench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HazyResearch/legalbench)   

- **[LEval](https://huggingface.co/datasets/L4NLP/LEval)** - Tasks: `Summarization`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FL4NLP%2FLEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/L4NLP/LEval) [![GitHub Stars](https://img.shields.io/github/stars/OpenLMLab/LEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenLMLab/LEval)   

- **[LexGLUE](https://github.com/coastalcph/lex-glue)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/coastalcph/lex-glue?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/coastalcph/lex-glue)   

- **[LEXTREME](https://github.com/JoelNiklaus/LEXTREME)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/JoelNiklaus/LEXTREME?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/JoelNiklaus/LEXTREME)   

- **[lila](https://huggingface.co/datasets/allenai/lila)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Flila&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/lila) [![GitHub Stars](https://img.shields.io/github/stars/allenai/Lila?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/Lila)   

- **[LMentry](https://github.com/aviaefrat/lmentry)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/aviaefrat/lmentry?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/aviaefrat/lmentry)   

- **[LMExamQA](https://huggingface.co/datasets/LEXam-Benchmark/LEXam)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FLEXam-Benchmark%2FLEXam&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/LEXam-Benchmark/LEXam)   

- **[LogiQA](https://github.com/lgw863/LogiQA-dataset)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/lgw863/LogiQA-dataset?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lgw863/LogiQA-dataset)   

- **[LongBench](https://huggingface.co/datasets/THUDM/LongBench)** - Tasks: `Question Answering`, `Reasoning`, `Summarization` | Mod: `Text` | Lang: `English`, `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTHUDM%2FLongBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/THUDM/LongBench) [![GitHub Stars](https://img.shields.io/github/stars/THUDM/LongBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/LongBench)   

- **[LongEval](https://github.com/DachengLi1/LongChat)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/DachengLi1/LongChat?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/DachengLi1/LongChat)   

- **[LooGLE](https://huggingface.co/datasets/bigainlco/LooGLE)** - Tasks: `Question Answering`, `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigainlco%2FLooGLE&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigainlco/LooGLE) [![GitHub Stars](https://img.shields.io/github/stars/bigai-nlco/LooGLE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/bigai-nlco/LooGLE)   

- **[M3KE](https://huggingface.co/datasets/TJUNLP/M3KE)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTJUNLP%2FM3KE&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TJUNLP/M3KE) [![GitHub Stars](https://img.shields.io/github/stars/tjunlp-lab/M3KE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tjunlp-lab/M3KE)   

- **[MCS-Bench](https://github.com/SCUT-DLVCLab/MCS-Bench)** - Tasks: `VQA`, ` Question Answering` | Mod: `Text`, `Video` | Lang: `Chinese` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FND-25%2FMCS-bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ND-25/MCS-bench)   [![GitHub Stars](https://img.shields.io/github/stars/SCUT-DLVCLab/MCS-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SCUT-DLVCLab/MCS-Bench) 

- **[MCTest](https://huggingface.co/datasets/sagnikrayc/mctest)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsagnikrayc%2Fmctest&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/sagnikrayc/mctest)   

- **[MCTS](https://github.com/blcuicall/mcts)** - Tasks: `Text Classification`, `Summarization` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/blcuicall/mcts?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/blcuicall/mcts)   

- **[MGSM](https://huggingface.co/datasets/juletxara/mgsm)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fjuletxara%2Fmgsm&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/juletxara/mgsm) [![GitHub Stars](https://img.shields.io/github/stars/google-research/url-nlp?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research/url-nlp)   

- **[MiniF2F_v1](https://github.com/openai/miniF2F)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/openai/miniF2F?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/miniF2F)   

- **[MLQA](https://huggingface.co/datasets/facebook/mlqa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmlqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/mlqa) [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/MLQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)]([![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ffacebook%2Fmlqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/facebook/mlqa))   

- **[MMCU](https://github.com/Felixgithub2017/MMCU)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/Felixgithub2017/MMCU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Felixgithub2017/MMCU)   

- **[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTIGER-Lab%2FMMLU-Pro&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) [![GitHub Stars](https://img.shields.io/github/stars/TIGER-AI-Lab/MMLU-Pro?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TIGER-AI-Lab/MMLU-Pro)   

- **[mmlu-redux-2.0](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fedinburgh-dawg%2Fmmlu-redux-2.0&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0) [![GitHub Stars](https://img.shields.io/github/stars/aryopg/mmlu-redux?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/aryopg/mmlu-redux)   

- **[MMMLU](https://huggingface.co/datasets/openai/MMMLU)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fopenai%2FMMMLU&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/openai/MMMLU) [![GitHub Stars](https://img.shields.io/github/stars/hendrycks/test?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hendrycks/test)   

- **[mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Flmsys%2Fmt_bench_human_judgments&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) [![GitHub Stars](https://img.shields.io/github/stars/lm-sys/FastChat?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)   

- **[NAH (Needle-in-a-Haystack)](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/gkamradt/LLMTest_NeedleInAHaystack?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)   

- **[NeedleBench 2024-7](https://github.com/open-compass/opencompass)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Englist`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/open-compass/opencompass?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/open-compass/opencompass)   

- **[NeuLR](https://github.com/deepreasoning/neulr)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/deepreasoning/neulr?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/deepreasoning/neulr)   

- **[OlympiadBench](https://github.com/OpenBMB/OlympiadBench)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/OpenBMB/OlympiadBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenBMB/OlympiadBench)   

- **[Owl-Bench](https://github.com/HC-Guo/Owl)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/HC-Guo/Owl?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HC-Guo/Owl)   

- **[PandaLM_testset](https://github.com/WeOpenML/PandaLM)** - Tasks: `Human Preference` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/WeOpenML/PandaLM?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/WeOpenML/PandaLM)   

- **[PAWS](https://github.com/google-research-datasets/paws)** - Tasks: `Semantic Matching` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/paws?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/paws)   

- **[PAWS-X](https://github.com/google-research-datasets/paws/tree/master/pawsx)** - Tasks: `Semantic Matching` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/google-research-datasets/paws?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research-datasets/paws/tree/master/pawsx)   

- **[PersianMMLU 2024-4](https://huggingface.co/datasets/raia-center/khayyam-challenge)** - Tasks: `Question Answering`, `Text Classification` | Mod: `Text` | Lang: `Persian` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fraia-center%2Fkhayyam-challenge&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/raia-center/khayyam-challenge)   

- **[PROST](https://huggingface.co/datasets/corypaik/prost)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcorypaik%2Fprost&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/corypaik/prost) [![GitHub Stars](https://img.shields.io/github/stars/nala-cub/prost?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nala-cub/prost)   

- **[QASPER](https://huggingface.co/datasets/allenai/qasper)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fqasper&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/qasper)   

- **[QiZhenGPT_eval](https://github.com/CMKRG/QiZhenGPT)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CMKRG/QiZhenGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CMKRG/QiZhenGPT)   

- **[QuAIL](https://huggingface.co/datasets/textmachinelab/quail)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftextmachinelab%2Fquail&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/textmachinelab/quail)   

- **[QuaRel](https://huggingface.co/datasets/community-datasets/quarel)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcommunity-datasets%2Fquarel&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/community-datasets/quarel)   

- **[raft](https://huggingface.co/datasets/ought/raft)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fought%2Fraft&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ought/raft)   

- **[RealTime QA](https://github.com/realtimeqa/realtimeqa_public)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/realtimeqa/realtimeqa_public?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/realtimeqa/realtimeqa_public)   

- **[ReClor](https://huggingface.co/datasets/metaeval/reclor)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmetaeval%2Freclor&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/metaeval/reclor)   

- **[SCALE](https://huggingface.co/collections/rcds/scale-datasets)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Frcds%2Fswiss_legislation&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/rcds/swiss_legislation)   

- **[SCIBENCH](https://huggingface.co/datasets/xw27/scibench)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fxw27%2Fscibench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/corypaik/prost) [![GitHub Stars](https://img.shields.io/github/stars/mandyyyyii/scibench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mandyyyyii/scibench)   

- **[SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fhicai-zju%2FSciKnowEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/hicai-zju/SciKnowEval) [![GitHub Stars](https://img.shields.io/github/stars/hicai-zju/sciknoweval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hicai-zju/sciknoweval)   

- **[SIGHAN](https://github.com/NYCU-NLP/SIGHAN-CSC)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/NYCU-NLP/SIGHAN-CSC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/NYCU-NLP/SIGHAN-CSC)   

- **[Social IQa](https://huggingface.co/datasets/allenai/social_i_qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fallenai%2Fsocial_i_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/social_i_qa)   

- **[SocKET](https://huggingface.co/datasets/Blablablab/SOCKET)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FBlablablab%2FSOCKET&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Blablablab/SOCKET) [![GitHub Stars](https://img.shields.io/github/stars/minjechoi/SOCKET?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/minjechoi/SOCKET)   

- **[SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Frajpurkar%2Fsquad_v2&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/rajpurkar/squad_v2)   

- **[STRATEGYQA](https://github.com/eladsegal/strategyqa)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/eladsegal/strategyqa?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/eladsegal/strategyqa)   

- **[SuperCLUE-Safety](https://github.com/CLUEbenchmark/SuperCLUE-safety)** - Tasks: `Instruction-Following`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/CLUEbenchmark/SuperCLUE-safety?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/CLUEbenchmark/SuperCLUE-safety)   

- **[SuperGLUE](https://huggingface.co/datasets/aps/super_glue)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Faps%2Fsuper_glue&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/aps/super_glue)   

- **[SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fm-a-p%2FSuperGPQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/m-a-p/SuperGPQA) [![GitHub Stars](https://img.shields.io/github/stars/SuperGPQA/SuperGPQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SuperGPQA/SuperGPQA)   

- **[TableBench](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMultilingual-Multimodal-NLP%2FTableBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench) [![GitHub Stars](https://img.shields.io/github/stars/TableBench/TableBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TableBench/TableBench)   

- **[tmmluplus](https://huggingface.co/datasets/ikala/tmmluplus)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fikala%2Ftmmluplus&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ikala/tmmluplus)   

- **[ToolEyes](https://github.com/Junjie-Ye/ToolEyes)** - Tasks: `Tool-Use` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Junjie-Ye/ToolEyes?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Junjie-Ye/ToolEyes)   

- **[TRUSTGPT](https://github.com/HowieHwong/TrustGPT)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/HowieHwong/TrustGPT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HowieHwong/TrustGPT)   

- **[UHGEval](https://huggingface.co/datasets/Ki-Seki/UHGEvalDataset)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FKi-Seki%2FUHGEvalDataset&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Ki-Seki/UHGEvalDataset) [![GitHub Stars](https://img.shields.io/github/stars/IAAR-Shanghai/UHGEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/IAAR-Shanghai/UHGEval)   

- **[WebQuestions](https://huggingface.co/datasets/stanfordnlp/web_questions)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fweb_questions&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/stanfordnlp/web_questions)   

- **[WenMind 2024-5](https://github.com/SCUT-DLVCLab/WenMind)** - Tasks: `Text Classification`, `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/SCUT-DLVCLab/WenMind?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/SCUT-DLVCLab/WenMind)   

- **[WiC](https://huggingface.co/datasets/sapienzanlp/wic)** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsapienzanlp%2Fwic&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/sapienzanlp/wic)   

- **[WikiEval](https://huggingface.co/datasets/explodinggradients/WikiEval)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fexplodinggradients%2FWikiEval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/explodinggradients/WikiEval) [![GitHub Stars](https://img.shields.io/github/stars/explodinggradients/ragas?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/explodinggradients/ragas)   

- **[WikiLingua](https://github.com/esdurmus/Wikilingua)** - Tasks: `Summarization` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/esdurmus/Wikilingua?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/esdurmus/Wikilingua)   

- **[WikiQA](https://huggingface.co/datasets/microsoft/wiki_qa)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwiki_qa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/microsoft/wiki_qa)   

- **[WinoGrande](https://huggingface.co/datasets/allenai/winogrande)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwinogrande&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/winogrande) [![GitHub Stars](https://img.shields.io/github/stars/allenai/winogrande?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/allenai/winogrande)   

- **[WinoWhy](https://github.com/HKUST-KnowComp/WinoWhy)** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/HKUST-KnowComp/WinoWhy?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/HKUST-KnowComp/WinoWhy)   

- **[WIQA](https://huggingface.co/datasets/allenai/wiqa)** - Tasks: `Reasoning`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwiqa&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/allenai/wiqa)   

- **[WritingBench](https://github.com/X-PLUG/WritingBench)** - Tasks: `Instruction-Following`, `Summarization` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/X-PLUG/WritingBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/X-PLUG/WritingBench)   

- **[WSC](https://huggingface.co/datasets/ErnestSDavis/winograd_wsc)** - Tasks: `Coreference Resolution` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwinograd_wsc&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ErnestSDavis/winograd_wsc)  

- **[WUNT2017](https://huggingface.co/datasets/leondz/wnut_17)** - Tasks: `Information Extraction` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fwnut_17&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/leondz/wnut_17)  

- **[WYWEB](https://github.com/baudzhou/WYWEB)** - Tasks: `Sequence Labeling`, `Sentence Classification`, `Token Similarity`, `Reading Comprehension`, `Translation` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/baudzhou/WYWEB?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/baudzhou/WYWEB)   

- **[XiezhiBenchmark](https://github.com/mikegu721/xiezhibenchmark)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/mikegu721/xiezhibenchmark?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mikegu721/xiezhibenchmark)   

- **[XNLI](https://github.com/facebookresearch/XNLI)** - Tasks: `Text Classification`, `Reasoning` | Mod: `Text` | Lang: `English`, `French`, `Spanish`, `German`, `Chinese`, `Russian`, `Arabic`, `Hindi`, `Portuguese`, `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/XNLI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/XNLI)   

- **[XSum](https://huggingface.co/datasets/EdinburghNLP/xsum)** - Tasks: `Summarization` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FEdinburghNLP%2Fxsum&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/EdinburghNLP/xsum) [![GitHub Stars](https://img.shields.io/github/stars/EdinburghNLP/XSum?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/EdinburghNLP/XSum)   

- **[XTREME](https://github.com/google-research/xtreme)** - Tasks: `Question Answering`, `Text Classification`, `Machine Translation` | Mod: `Text` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/google-research/xtreme?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research/xtreme)   

- **[YACLC](https://github.com/blcuicall/YACLC)** - Tasks: `Text Classification` | Mod: `Text` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/blcuicall/YACLC?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/blcuicall/YACLC)   

- **[Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fmultimodal-reasoning-lab%2FZebra-CoT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) [![GitHub Stars](https://img.shields.io/github/stars/multimodal-reasoning-lab/Bagel-Zebra-CoT?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT)   

- **[ZebraLogic](https://huggingface.co/datasets/WildEval/ZebraLogic)** - Tasks: `Reasoning` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FWildEval%2FZebraLogic&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/WildEval/ZebraLogic) [![GitHub Stars](https://img.shields.io/github/stars/WildEval/ZeroEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/WildEval/ZeroEval)   

- **[zero_scrolls](https://huggingface.co/datasets/tau/zero_scrolls)** - Tasks: `Summarization`, `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Ftau%2Fzero_scrolls&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/tau/zero_scrolls) [![GitHub Stars](https://img.shields.io/github/stars/tau-nlp/zero_scrolls?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/tau-nlp/zero_scrolls)   



<a id="text-retrieval-rag"></a>
#### Retrieval / RAG
- **[DragonBall](https://github.com/OpenBMB/RAGEval/tree/main/dragonball_dataset)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/OpenBMB/RAGEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenBMB/RAGEval)   

- **[RAG-Instruct-Benchmark-Tester](https://huggingface.co/datasets/llmware/rag_instruct_benchmark_tester)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fllmware%2Frag_instruct_benchmark_tester&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/llmware/rag_instruct_benchmark_tester)   

- **[RGB](https://github.com/chen700564/RGB)** - Tasks: `Question Answering` | Mod: `Text` | Lang: `English`, `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/chen700564/RGB?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/chen700564/RGB) | 

---

<a id="code"></a>
### Code


<a id="code-pretraining"></a>
#### Pretraining
- **[CodeParrot](https://huggingface.co/datasets/codeparrot/codeparrot-clean)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcodeparrot%2Fcodeparrot-clean&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/codeparrot/codeparrot-clean) 

- **[github-code](https://huggingface.co/datasets/codeparrot/github-code)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcodeparrot%2Fgithub-code&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/codeparrot/github-code) 

- **[starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigcode%2Fstarcoderdata&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigcode/starcoderdata)   

- **[the-stack](https://huggingface.co/datasets/bigcode/the-stack)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `English`, `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigcode%2Fthe-stack&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigcode/the-stack)  


<a id="code-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[APIBench](https://huggingface.co/datasets/gorilla-llm/APIBench)** - Tasks:  `Tool-Use` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fgorilla-llm%2FAPIBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/gorilla-llm/APIBench)   

- **[CodeAlpaca‑20K](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fsahil2801%2FCodeAlpaca-20k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) [![GitHub Stars](https://img.shields.io/github/stars/sahil280114/codealpaca?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/sahil280114/codealpaca) 

- **[instructional_codesearchnet_python](https://huggingface.co/datasets/Nan-Do/instructional_code-search-net-python)** - Tasks: `Instruction-Following`, `Code Generation` | Mod: `Code` | Lang:  `Python` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FNan-Do%2Finstructional_code-search-net-python&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Nan-Do/instructional_code-search-net-python)  

- **[Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)** - Tasks: `Instruction-Following`, `Code Generation`, `Code Completion` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fise-uiuc%2FMagicoder-OSS-Instruct-75K&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) [![GitHub Stars](https://img.shields.io/github/stars/ise-uiuc/magicoder?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/ise-uiuc/magicoder)  


<a id="code-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*


<a id="code-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- **[apps](https://huggingface.co/datasets/codeparrot/apps)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcodeparrot%2Fapps&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/codeparrot/apps) [![GitHub Stars](https://img.shields.io/github/stars/hendrycks/apps?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/hendrycks/apps)  

- **[Berkeley Function Calling Leaderboard (BFCL)](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)** - Tasks: `Tool-Use` | Mod: `Code` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fgorilla-llm%2FBerkeley-Function-Calling-Leaderboard&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)   

- **[BIRD](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)** - Tasks: `Question Answering`, `Code Generation` | Mod: `Text`, `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/AlibabaResearch/DAMO-ConvAI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) 

- **[CodeElo](https://huggingface.co/datasets/Qwen/CodeElo)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FQwen%2FCodeElo&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Qwen/CodeElo) [![GitHub Stars](https://img.shields.io/github/stars/QwenLM/CodeElo?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/QwenLM/CodeElo)   

- **[CodeXGLUE](https://github.com/microsoft/CodeXGLUE)** - Tasks: `Code Generation`, `Code Completion`, `Code Summarization` | Mod: `Code`, `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/microsoft/CodeXGLUE?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/microsoft/CodeXGLUE)  

- **[cruxeval](https://huggingface.co/datasets/cruxeval-org/cruxeval)** - Tasks: `Code Reasoning`,`Code Understanding` | Mod: `Code` | Lang: `Python` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fcruxeval-org%2Fcruxeval&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/cruxeval-org/cruxeval) [![GitHub Stars](https://img.shields.io/github/stars/facebookresearch/cruxeval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/facebookresearch/cruxeval)  

- **[CSpider](https://github.com/taolusi/chisp)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Chinese` | [![GitHub Stars](https://img.shields.io/github/stars/taolusi/chisp?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/taolusi/chisp) 

- **[DomainEval 2024-8](https://github.com/domaineval/DomainEval)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair`, `Code Summarization` | Mod: `Code` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/domaineval/DomainEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/domaineval/DomainEval)  

- **[DS-1000](https://github.com/xlang-ai/DS-1000)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/xlang-ai/DS-1000?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/xlang-ai/DS-1000)   

- **[HumanEval](https://github.com/openai/human-eval)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/openai/human-eval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/openai/human-eval)  

- **[HumanEval+ 2023-5](https://github.com/evalplus/evalplus)** - Tasks: `Code Generation`, `Code Repair` | Mod: `Code` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/evalplus/evalplus?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/evalplus/evalplus)  

- **[humanevalpack](https://huggingface.co/datasets/bigcode/humanevalpack)** - Tasks: `Code Generation`, `Code Completion`, `Code Repair` | Mod: `Code` | Lang: ``Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fbigcode%2Fhumanevalpack&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/bigcode/humanevalpack) [![GitHub Stars](https://img.shields.io/github/stars/bigcode-project/octopack?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/bigcode-project/octopack)  

- **[MBPP](https://github.com/google-research/google-research/tree/master/mbpp)** - Tasks: `Code Generation`, `Code Completion` | Mod: `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/google-research/google-research?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/google-research/google-research/tree/master/mbpp)    

- **[MTPB](https://github.com/salesforce/CodeGen)** - Tasks: `Code Generation` | Mod: `Code` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/salesforce/CodeGen?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/salesforce/CodeGen)  

- **[ODEX](https://github.com/zorazrw/odex)** - Tasks: `Code Generation` | Mod: `Text`, `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/zorazrw/odex?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/zorazrw/odex)  

<a id="code-retrieval-rag"></a>
#### Retrieval / RAG
- *(add entries)*

---

<a id="multimodal"></a>
### Multimodal


<a id="multimodal-pretraining"></a>
#### Pretraining
- **[MedTrinity-25M](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FUCSC-VLAA%2FMedTrinity-25M&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M) [![GitHub Stars](https://img.shields.io/github/stars/UCSC-VLAA/MedTrinity-25M?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/UCSC-VLAA/MedTrinity-25M)

- **[OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)** - Tasks: `Image Captioning`, `VQA`, `Information Extraction` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceM4%2FOBELICS&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)

- **[OBELISC](https://huggingface.co/datasets/HuggingFaceM4/OBELICS)** - Mod: `Text`, `Image` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FHuggingFaceM4%2FOBELICS&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) [![GitHub Stars](https://img.shields.io/github/stars/huggingface/OBELICS?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/huggingface/OBELICS)



<a id="multimodal-instruction-tuning"></a>
#### Instruction Tuning / SFT
- **[cc_sbu_align](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align)** - Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FVision-CAIR%2Fcc_sbu_align&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align)  

- **[InstructDoc](https://github.com/nttmdlab-nlp/InstructDoc)** - Tasks: `Instruction-Following`, `Information Extraction` | Mod: `Multimodel` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/nttmdlab-nlp/InstructDoc?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/nttmdlab-nlp/InstructDoc)

- **[JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB)** - Tasks: `Question Answering`, `Image Captioning` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FJourneyDB%2FJourneyDB&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/JourneyDB/JourneyDB) 

- **[LLaVA Visual Instruct 150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)** - Tasks: `Instruction-Following` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fliuhaotian%2FLLaVA-Instruct-150K&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) [![GitHub Stars](https://img.shields.io/github/stars/haotian-liu/LLaVA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/haotian-liu/LLaVA) 

- **[M3IT](https://huggingface.co/datasets/MMInstruction/M3IT)** - Tasks: `Instruction-Following` | Mod: `Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMMInstruction%2FM3IT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MMInstruction/M3IT) 

- **[MIMIC-IT](https://github.com/Luodian/Otter/tree/main/mimic-it)** - Tasks: `Instruction-Following`, `Video Understanding` | Mod: `Image`, `Video` | Lang: `Multi` | [![GitHub Stars](https://img.shields.io/github/stars/Luodian/Otter?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Luodian/Otter/tree/main/mimic-it) 

- **[ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA)** - Tasks: `Question Answering`, `Reasoning` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fderek-thomas%2FScienceQA&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/derek-thomas/ScienceQA) [![GitHub Stars](https://img.shields.io/github/stars/lupantech/ScienceQA?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/lupantech/ScienceQA) 

- **[ShareGPT4V](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V)** - Tasks: `Image Captioning` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FLin-Chen%2FShareGPT4V&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V) 

- **[VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT)** - Tasks: `Instruction-Following`, `Dialogue` | Mod: `Text`, `Video` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenGVLab%2FVideoChat2-IT&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) 



<a id="multimodal-alignment-rlhf"></a>
#### Alignment / RL



<a id="multimodal-evaluation-benchmark"></a>
#### Evaluation / Benchmark
- **[ALM-Bench](https://huggingface.co/datasets/MBZUAI/ALM-Bench)** - Mod: `Text`, `Image`, `Video`, `Audio` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMBZUAI%2FALM-Bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MBZUAI/ALM-Bench) [![GitHub Stars](https://img.shields.io/github/stars/mbzuai-oryx/ALM-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mbzuai-oryx/ALM-Bench) 

- **[II-Bench](https://huggingface.co/datasets/m-a-p/II-Bench)** - Mod: `Image`,`Text` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fm-a-p%2FII-Bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/m-a-p/II-Bench) [![GitHub Stars](https://img.shields.io/github/stars/II-Bench/II-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/II-Bench/II-Bench) 

- **[M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR)** | - Tasks: `Information Retrieval` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FTIGER-Lab%2FM-BEIR&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) [![GitHub Stars](https://img.shields.io/github/stars/TIGER-AI-Lab/UniIR?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/TIGER-AI-Lab/UniIR)

- **[MM-NIAH](https://huggingface.co/datasets/OpenGVLab/MM-NIAH)** - Tasks: `Information Extraction` | Mod: `Text`, `image` | Lang: `English` |[![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenGVLab%2FMM-NIAH&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenGVLab/MM-NIAH) [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/MM-NIAH?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenGVLab/MM-NIAH)  

- **[MME-RealWorld](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld)** - Tasks: `Question Answering`, `Image Captioning` | Mod: `Text`, `image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fyifanzhang114%2FMME-RealWorld&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld) [![GitHub Stars](https://img.shields.io/github/stars/yfzhang114/MME-RealWorld?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/yfzhang114/MME-RealWorld)  

- **[MMIU](https://huggingface.co/datasets/FanqingM/MMIU-Benchmark)** - Tasks: `Image Understanding` | Mod: `Image`,`Text` | Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FFanqingM%2FMMIU-Benchmark&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/FanqingM/MMIU-Benchmark) [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/MMIU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenGVLab/MMIU) 

- **[MMMU](https://huggingface.co/datasets/MMMU/MMMU)** - Tasks: `VQA`, `Reasoning` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FMMMU%2FMMMU&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/MMMU/MMMU) [![GitHub Stars](https://img.shields.io/github/stars/MMMU-Benchmark/MMMU?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/MMMU-Benchmark/MMMU)  

- **[MMT-Bench](https://huggingface.co/datasets/Kaining/MMT-Bench)** - Tasks: `VQA`, `Reasoning` | Mod: `Text`, `Image` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FKaining%2FMMT-Bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Kaining/MMT-Bench) [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/MMT-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenGVLab/MMT-Bench)  

- **[MRAG-Bench](https://huggingface.co/datasets/uclanlp/MRAG-Bench)** | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2Fuclanlp%2FMRAG-Bench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/uclanlp/MRAG-Bench) [![GitHub Stars](https://img.shields.io/github/stars/mragbench/MRAG-Bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/mragbench/MRAG-Bench) 

- **[MultiTrust](https://github.com/thu-ml/MMTrustEval)** - Tasks: `VQA` | Mod: `Text`, `Image` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/thu-ml/MMTrustEval?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/thu-ml/MMTrustEval)  

- **[MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench)** - Tasks: `Video Understanding` | Mod: `Video`,`Text`| Lang: `Multi` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FOpenGVLab%2FMVBench&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/OpenGVLab/MVBench) [![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/Ask-Anything?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2) 

- **[PromptBench](https://github.com/microsoft/promptbench)** - Tasks: `Multi` | Mod: `Text`, `Image` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/microsoft/promptbench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/microsoft/promptbench)  




<a id="multimodal-retrieval-rag"></a>
#### Retrieval / RAG
- **ViDoRe** - Tasks: `Question Answering`, `VQA` | Mod: `Text`, `Image`, `Video` | Lang: `English`

---

<a id="gen"></a>
### Generation (Image/Video/Audio)


<a id="gen-pretraining"></a>
#### Pretraining
- **[Dataset-E](link)** — Tags: `GeneralLM`, `Image-Text`, `English` — Pretraining for image generation…


<a id="gen-instruction-tuning"></a>
#### Instruction Tuning / SFT


<a id="gen-alignment-rlhf"></a>
#### Alignment / RL
- *(add entries)*


<a id="gen-evaluation-benchmark"></a>
#### Evaluation / Benchmark



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
- **[xLAM Function Calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)** - Tasks: `Tool-Use` | Mod: `Code` | Lang: `English` | [![HF](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FSalesforce%2Fxlam-function-calling-60k&query=%24.downloads&label=HF&labelColor=0d1117&color=2ecc71&logo=huggingface&logoColor=ffcc00)](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)


<a id="agent-instruction-tuning"></a>
#### Alignment / RL
- *(add entries)*


<a id="agent-evaluation-benchmark"></a>
#### Evaluation / Benchmark

- **[AgentBench](https://github.com/THUDM/AgentBench)** - Tasks: `Reasoning`, `Planning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/THUDM/AgentBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/THUDM/AgentBench) 

- **[API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)** - Tasks: `Dialogue`, `Question Answering`, `Tool-Use`, `Planning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/AlibabaResearch/DAMO-ConvAI?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) 

- **[GameBench 2024-6](https://github.com/Joshuaclymer/GameBench)** - Tasks: `Reasoning`, `Planning` | Mod: `Text` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/Joshuaclymer/GameBench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/Joshuaclymer/GameBench) 

- **[MINT](https://github.com/xingyaoww/mint-bench)** - Tasks: `Tool-Use`, `Reasoning` | Mod: `Text`, `Code` | Lang: `English` | [![GitHub Stars](https://img.shields.io/github/stars/xingyaoww/mint-bench?style=flat&label=Stars&labelColor=0d1117&color=ea4aaa&logo=github&logoColor=white)](https://github.com/xingyaoww/mint-bench) 

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


