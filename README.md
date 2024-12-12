<h1 align="center">
  Stanceformer
</h1>

<h4 align="center">A Target-Aware Transformer Model for Stance Detection</h4>

<p align="center">
  <a href="https://aclanthology.org/2024.findings-emnlp.286/"><img src="https://img.shields.io/badge/Findings%20of%20EMNLP-2024-red"></a>
  <a href="https://aclanthology.org/2024.findings-emnlp.286.pdf"><img src="https://img.shields.io/badge/Paper-PDF-yellow"></a>
  <a href="https://github.com/kgarg8/Stanceformer/blob/master/res/Stanceformer.pdf"><img src="https://img.shields.io/badge/Presentation-PDF-blue"></a>
  <a href="https://github.com/kgarg8/Stanceformer/blob/master/res/Stanceformer_Poster.pdf"><img src="https://img.shields.io/badge/Poster-PDF-blue"></a>
  <a href="https://github.com/kgarg8/Stanceformer/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

## Abstract
The task of Stance Detection involves discerning the stance expressed in a text towards a specific subject or target. Prior works have relied on existing transformer models that lack the capability to prioritize targets effectively. Consequently, these models yield similar performance regardless of whether we utilize or disregard target information, undermining the task’s significance. 

To address this challenge, we introduce **Stanceformer**, a target-aware transformer model that incorporates enhanced attention towards the targets during both training and inference. Specifically, we design a Target Awareness matrix that increases the self-attention scores assigned to the targets. 

We demonstrate the efficacy of Stanceformer with various BERT-based models, including state-of-the-art models and Large Language Models (LLMs), and evaluate its performance across three stance detection datasets, alongside a zero-shot dataset. Our approach not only provides superior performance but also generalizes to other domains, such as Aspect-based Sentiment Analysis.

## Features
- Target-aware attention mechanism
- Enhanced performance on stance detection tasks
- Generalization to other NLP domains
