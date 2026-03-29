i am working on it; it's incomlete 
# Neural Deep Learning Suite

A research-grade modular deep learning framework built from scratch to implement, train, analyze, and benchmark multiple neural network architectures across different paradigms.

---

## 🚀 Overview

This project is not just about training models — it is a **complete deep learning system** that includes:

- From-scratch implementations of core architectures
- Modular training and evaluation pipeline
- Config-driven experiment management
- Benchmarking and ablation system
- Visualization and interpretability tools

The goal is to demonstrate **deep understanding of ML systems, not just usage of libraries**.

---

## 🧠 Architectures Implemented

### 1. CNNs
- ResNet-18 (from scratch)
- Residual blocks and skip connections

### 2. Sequence Models
- Custom LSTM Cell (manual implementation)
- Attention mechanism (Bahdanau)
- Seq2Seq architecture

### 3. Transformers
- Vision Transformer (ViT)
- Patch embeddings
- Multi-head self-attention

### 4. Graph Neural Networks
- GCN (Graph Convolutional Network)
- GraphSAGE

---

## 🏗️ Project Structure
neural-deep-learning-suite/
│
├── core/ # Training engine
│ ├── trainer.py
│ ├── evaluator.py
│ ├── metrics.py
│ ├── hooks.py
│ ├── checkpoint.py
│ ├── utils.py
│
├── data/ # Data handling
│ ├── datasets.py
│ ├── transforms.py
│ ├── dataloader.py
│
├── models/ # Model implementations
│ ├── resnet/
│ ├── lstm_attention/
│ ├── vit/
│ ├── gnn/
│ ├── diffusion/ 
│
├── interpretability/ # Model analysis tools
│ ├── gradcam.py
│ ├── shap.py
│ ├── probing.py
│ ├── loss_landscape.py
│
├── benchmark/ # Experiment system
│ ├── runner.py
│ ├── ablations.py
│
├── visualization/ # Plots & graphs
│ ├── plot_metrics.py
│ ├── plot_comparisons.py
│ ├── plot_lr_schedule.py
│
├── configs/ # Experiment configs
│ ├── default.yaml
│ ├── resnet.yaml
│ ├── vit.yaml
│ ├── lstm.yaml
│ ├── gnn.yaml
│ ├── ablations/
│
├── experiments/ # Entry points
│ ├── resnet_train.py
│ ├── vit_train.py
│ ├── lstm_train.py
│ ├── gnn_train.py
│
├── tests/ # Validation tests
│ ├── test_overfit.py
│ ├── test_shapes.py
│
├── logs/
├── checkpoints/
├── README.md
└── requirements.txt


---

## ⚙️ Key Features

### 🔹 Modular Training System
- Custom Trainer with hooks system
- Supports schedulers, checkpointing, and extensions

### 🔹 Config-Driven Experiments
- YAML-based configuration
- Reproducible and scalable experiments

### 🔹 Multi-Paradigm Support
- Vision (CNN, ViT)
- Sequence (LSTM + Attention)
- Graphs (GNN)

### 🔹 Benchmarking Engine
- Compare models across metrics
- Run ablation studies
- Measure trade-offs (accuracy vs speed)

### 🔹 Visualization Layer
- Training curves
- Model comparison plots
- Learning rate schedules

### 🔹 Interpretability Tools
- Grad-CAM for CNNs
- SHAP integration
- Probing intermediate representations
- Loss landscape analysis

---

## 🧪 Experiments

### Run a Model

```bash
python experiments/resnet_train.py

Or using config system:

run("configs/resnet.yaml")