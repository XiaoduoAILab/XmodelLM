<h1 align="center">
Xmodel_LM-1.1B
</h1>

<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Xiaoduo%20HuggingFace-blue.svg)](https://huggingface.co/XiaoduoAILab/Xmodel_LM)
[![arXiv](https://img.shields.io/badge/Arxiv-2406.02856-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2406.02856) 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/XiaoduoAILab/XmodelLM.git)[![github](https://img.shields.io/github/stars/XiaoduoAILab/XmodelLM.svg?style=social)](https://github.com/XiaoduoAILab/XmodelLM.git)  


</h5>

## 🌟 Introduction

We introduce Xmodel-LM, a compact and efficient 1.1B language model pre-trained on around 2 trillion tokens. Trained on our self-built dataset (Xdata), which balances Chinese and English corpora based on downstream task optimization, Xmodel-LM exhibits remarkable performance despite its smaller size. It notably surpasses existing open-source language models of similar scale.

## 📊 Benchmark

### Commonsense Reasoning

| Model | ARC-c | ARC-e | Boolq | HellaSwag | OpenbookQA | PiQA | SciQ | TriviaQA | Winogrande | Avg |
|-------|-------|-------|-------|-----|-----|------|------|-----|-------|-----|
| OPT-1.3B | 23.29 | 57.03 | 57.80 | 41.52 | 23.20 | 71.71 | 84.30 | 7.48 | 59.59 | 47.32 |
| Pythia-1.4B | 25.60 | 57.58 | 60.34 | 39.81 | 20.20 | 71.06 | 85.20 | 5.01 | 56.20 | 47.00 |
| TinyLLaMA-3T-1.1B | 27.82 | 60.31 | 57.83 | 44.98 | 21.80 | 73.34 | 88.90 | 11.30 | 59.12 | 48.59 |
| MobileLLaMA-1.4B | 26.28 | 61.32 | 57.92 | 42.87 | 23.60 | 71.33 | 87.40 | 12.02 | 58.25 | 49.00 |
| Qwen1.5-1.8B | 32.25 | 64.69 | 66.48 | 45.49 | 23.80 | 73.45 | 92.90 | 1.01 | 61.17 | 51.25 |
| **Xmodel-LM-1.1B** | 28.16 | 62.29 | 61.44 | 45.96 | 24.00 | 72.03 | 89.70 | 18.46 | 60.62 | 51.41 |
| H2O-danube-1.8B | 32.94 | 67.42 | 65.75 | 50.85 | 27.40 | 75.73 | 91.50 | 25.05 | 62.35 | 55.44 |
| InternLM2-1.8B | 37.54 | 70.20 | 69.48 | 46.52 | 24.40 | 75.57 | 93.90 | 36.67 | 65.67 | 57.77 |


### Problem Solving

| Model | BBH (3-shot) | GLUE (5-shot) | GSM8K (5-shot) | MMLU (5-shot) | Avg | Avg w.o. GSM8k |
|-------|-----|------|-------|------|-----|----------------|
| OPT-1.3B | 22.67 | 51.06 | 0.83  | 26.70 | 25.32 | 33.48         |
| Pythia-1.4B | 25.37 | 52.23 | 1.63  | 25.40 | 26.16 | 34.33         |
| MobileLLaMA-1.4B | 23.48 | 43.34 | 1.44  | 24.60 | 23.22 | 30.47         |
| TinyLLaMA-3T-1.1B | 26.75 | 48.25 | 1.97  | 25.70 | 25.67 | 33.57         |
| H2O-danube-1.8B | 27.31 | 49.83 | 1.90  | 25.70 | 26.19 | 34.28         |
|  **Xmodel-LM-1.1B** | 27.34 | 52.61 | 2.58 | 25.90 | 27.11 | 35.28         |
| InternLM2-1.8B | 16.86 | 58.96 | 23.50 | 42.00 | 35.34 | 39.27         |
| Qwen1.5-1.8B | 13.84 | 64.57 | 33.59 | 45.10 | 39.28 | 41.17         |


### Chinese Ability

| Model | ARC-zh | XCOPA-zh | XNLI-zh | Avg |
|-------|--------|-----------|----------|-----|
| OPT-1.3B | 18.80  | 53.00     | 33.45    | 35.08|
| Pythia-1.4B | 21.03  | 52.60     | 34.06    | 35.90|
| MobileLLaMA-1.4B | 20.26  | 52.80     | 33.82    | 35.63|
| TinyLLaMA-3T-1.1B | 21.37  | 56.80     | 33.25    | 37.14|
| H2O-danube-1.8B | 21.79  | 55.60     | 34.74    | 37.38|
| **Xmodel-LM-1.1B** | 26.24  | 60.60     | 36.02    | 40.95|
| InternLM2-1.8B | 27.69  | 66.80     | 34.58    | 43.00|
| Qwen1.5-1.8B | 32.14  | 66.00     | 39.28    | 45.81|


## 🛠️ Install

1. Clone this repository and navigate to XmodelLM folder
   ```bash
   git clone https://github.com/XiaoduoAILab/XmodelLM.git
   cd xmodellm
   ```

2. Install Package
    ```Shell
    pip install -r requirements.txt
    ```

## 🗝️ Quick Start

#### Download Xmodel_LM model

Our model files are fully open source on huggingface, you can download them at [here](https://huggingface.co/XiaoduoAILab/Xmodel_LM).

#### Example for Xmodel_LM model inference
You need to download the model files first and save them in your folder. Then you can run the scripts below.
```bash
python generate.py --model_path path/to/folder --device cuda:0
```

## ✏️ Reference

If you find Xmodel_LM useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:

```
@misc{wang2024xmodellm,
      title={Xmodel-LM Technical Report}, 
      author={Yichuan Wang and Yang Liu and Yu Yan and Xucheng Huang and Ling Jiang},
      year={2024},
      eprint={2406.02856},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

