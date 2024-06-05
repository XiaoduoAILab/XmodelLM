<h1 align="center">
Xmodel_LM-1.1B
</h1>

<h5 align="center">

[![hf_space](https://img.shields.io/badge/🤗-Xiaoduo%20HuggingFace-blue.svg)](https://huggingface.co/XiaoduoAILab/Xmodel_LM)
[![arXiv](https://img.shields.io/badge/Arxiv-2405.09215-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.09215) 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/XiaoduoAILab/XmodelLM.git)[![github](https://img.shields.io/github/stars/XiaoduoAILab/XmodelLM.svg?style=social)](https://github.com/XiaoduoAILab/XmodelLM.git)  


</h5>

## 🌟 Introduction

We introduce Xmodel-LM, a compact and efficient 1.1B language model pre-trained on over 2 trillion tokens. Trained on our self-built dataset (Xdata), which balances Chinese and English corpora based on downstream task optimization, Xmodel-LM exhibits remarkable performance despite its smaller size. It notably surpasses existing open-source language models of similar scale.

## 📊 Benchmark

### Commonsense Reasoning

| Model | ARC-c | ARC-e | Boolq | HS. | OB. | PiQA | SciQ | TQ. | Wino. | Avg |
|-------|-------|-------|-------|-----|-----|------|------|-----|-------|-----|
|       |       |       |       |     |     |      |      |     |       |     |
| OPT-1.3B | 23.29 | 57.03 | 57.80 | 41.52 | 23.20 | 71.71 | 84.30 | 7.48 | 59.59 | 47.32 |
| Pythia-1.4B | 25.60 | 57.58 | 60.34 | 39.81 | 20.20 | 71.06 | 85.20 | 5.01 | 56.20 | 47.00 |
| TinyLLaMA-3T-1.1B | 27.82 | 60.31 | 57.83 | 44.98 | 21.80 | 73.34 | 88.90 | 11.30 | 59.12 | 48.59 |
| MobileLLaMA-1.4B | 26.28 | 61.32 | 57.92 | 42.87 | 23.60 | 71.33 | 87.40 | 12.02 | 58.25 | 49.00 |
| Qwen1.5-1.8B | 32.25 | 64.69 | 66.48 | 45.49 | 23.80 | 73.45 | 92.90 | 1.01 | 61.17 | 51.25 |
| H2O-danube-1.8B | 32.94 | 67.42 | 65.75 | 50.85 | 27.40 | 75.73 | 91.50 | 25.05 | 62.35 | 55.44 |
| InternLM2-1.8B | 37.54 | 70.20 | 69.48 | 46.52 | 24.40 | 75.57 | 93.90 | 36.67 | 65.67 | 57.77 |
| Xmodel-1.1B | 26.54 | 61.20 | 60.40 | 45.44 | 24.40 | 71.55 | 89.50 | 16.18 | 59.43 | 50.52 |

### 

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

#### Example for Xmodel_LM model inference
```bash
python generate.py
```

