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

