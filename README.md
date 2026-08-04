---
title: Medical Chatbot
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
---

# 🩺 AI Medical Symptom Checker Chatbot

[![HuggingFace Spaces](https://img.shields.io/badge/Live_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/dharaamehta33/medical-qa-chatbot)
[![Model](https://img.shields.io/badge/Model-HuggingFace_Hub-blue)](https://huggingface.co/dharaamehta33/medical-chatbot-llama)

## What I Built
Fine-tuned Llama 3.2 (1B) on medical Q&A pairs using Unsloth + QLoRA.
Deployed as a live Gradio web app on HuggingFace Spaces.

## Tech Stack
- **Model:** Meta Llama 3.2 1B Instruct
- **Fine-tuning:** Unsloth + QLoRA (4-bit quantization)
- **Dataset:** MedQuAD (`lavita/MedQuAD`) — 2,000 examples
- **UI:** Gradio
- **Deployment:** HuggingFace Spaces (free)
- **GPU:** Google Colab T4 (free)
