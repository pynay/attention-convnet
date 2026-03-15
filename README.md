# Attention-Augmented CNN for Image Classification

Investigating whether lightweight self-attention modules improve CNN-based image classification on CIFAR-10 and CIFAR-100.

## Overview

We implement a VGG-style baseline CNN and an attention-augmented variant that inserts spatial self-attention or channel attention (SE-block) after configurable convolutional blocks. Through ablation studies, we compare attention type, placement, and the effect across datasets.

**Key results:**
| Configuration | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Baseline CNN | 92.29% | 69.44% |
| Spatial attn [0,1,2] | **92.70%** | 68.92% |
| Channel attn [1,2] | 92.62% | — |

## Project Structure

```
├── src/
│   ├── attention.py    # Spatial self-attention & channel attention (SE-block)
│   ├── models.py       # BaselineCNN & AttentionCNN architectures
│   └── train.py        # Training pipeline, data loading, evaluation
├── notebooks/
│   └── experiments.ipynb  # All experiments, ablations, and visualizations
├── results/            # Saved checkpoints, metrics JSON, and plots
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Train from command line:**
```bash
# Baseline
python -m src.train --model-type baseline

# Attention-augmented (spatial attention after blocks 1 and 2)
python -m src.train --model-type attention --attention-type spatial --attention-positions 1 2

# CIFAR-100
python -m src.train --dataset cifar100 --model-type attention --attention-positions 0 1 2
```

**Run all experiments:** Open `notebooks/experiments.ipynb` and run all cells. Existing results are loaded from `results/` automatically; only missing configurations are trained.

## Method

- **Baseline:** 4-block CNN (3→64→128→256→256), global avg pool, FC classifier (~2.36M params)
- **Spatial attention:** Multi-head scaled dot-product attention over spatial positions with residual connection and LayerNorm
- **Channel attention:** Squeeze-and-excitation with bottleneck MLP (reduction ratio 16)
- **Training:** Adam (lr=1e-3, weight decay=1e-4), cosine annealing, 50 epochs, batch size 128
