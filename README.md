# Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model

**CVPR 2026**

[Dongwon Kim](https://kdwonn.github.io)<sup>1</sup>, [Gawon Seo](https://www.linkedin.com/in/gawon-seo-9a3588279/)<sup>2</sup>, [Jinsung Lee](https://jinsingsangsung.github.io/)<sup>2</sup>, [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/)<sup>2,3</sup>, [Suha Kwak](https://suhakwak.github.io/)<sup>2</sup>

<sup>1</sup>KAIST &nbsp; <sup>2</sup>POSTECH &nbsp; <sup>3</sup>RLWRLD

[[Project Page](https://kdwonn.github.io/CompACT)] [[Paper](https://arxiv.org/abs/2603.05438)]

## Overview

This repo consists of two main training pipelines:
1. **Tokenizer training**: Train CompACT, compact tokenizer that compress image up to 8 discrete tokens
2. **World Model Training**: Trains the CDiT model using the learned tokenizers for navigation prediction


## Quick Start

This project uses `uv` for dependency management. Simply run:

```bash
# Install dependencies (first time only)
uv sync

# Run any script
uv run <script>

# Examples:
uv run train_tokenizer.py  # Train tokenizer
uv run bash scripts/train.sh --nproc=4 -- ++tokenizer_path=<TOKENIZER_DIR>  # Train world model
```

## Data & Checkpoint Preparation

- Requires ImageNet in webdataset format for tokenizer training. For navigation dataset, see original [NWM repo](https://github.com/facebookresearch/nwm/).
- Download following checkpoints: [MAGE](https://github.com/LTH14/mage), [DINOv3](https://github.com/facebookresearch/dinov3)
- Set following environment variables for ckpt and dataset dirs: `DATASET_PREFIX`, `BASE_TOKENIZER_CKPT`

## Architecture

### Tokenizer Training Pipeline

```bash
# Train tokenizer with default configuration
uv run train_tokenizer.py

# With custom image size (default: 224)
uv run train_tokenizer.py ++dataset.image_size=256 ++dataset.dinov2_image_size=256
```

### World Model Training Pipeline

The main CDiT model supports diverse tokenizer architectures and is trained on navigation data:

#### Supported Tokenizers

The codebase supports multiple tokenizer types (`conf/model/tokenizer/`):
- **CompactTok**: Efficient tokenization with extreme compression (proposed)
- **FlexTok**: Flexible tokenization with variable token length
- **SDVAE**: Stable Diffusion VAE-based tokenization

```bash
# Train world model (requires tokenizer checkpoint)
uv run bash scripts/train.sh --nproc=4 -- ++tokenizer_path=<TOKENIZER_DIR>
```

## Configuration System

The project uses [Hydra](https://hydra.cc/) for configuration management with two separate configuration directories:

### Main Training Configuration (`conf/`)
```
conf/
├── config.yaml              # Main configuration file
├── model/
│   ├── generator/          # CDiT model configurations (B, B_disc, L_disc, XL)
│   ├── tokenizer/          # Tokenizer configurations
│   └── diffusion/          # Diffusion configurations (gaussian, discrete, discrete_history)
├── training/               # Training configurations
├── dataset/                # Dataset configurations
└── scheduler/              # LR scheduler configurations
```

### Tokenizer Training Configuration (`conf_tokenizer/`)
```
conf_tokenizer/
├── config.yaml             # Tokenizer training configuration
├── model/
│   ├── compact_tokenizer/  # CompactTokenizer (DINOv3, ViT-based, RIN) configs
│   ├── base_tokenizer/     # Base tokenizer (MAGE, OpenMAGViT) configs
│   ├── diffusion/          # Discrete diffusion configs
│   └── rep_guidance/       # Representation guidance loss configs
├── training/               # Training configurations
├── dataset/                # Dataset configs (ImageNet, etc.)
└── scheduler/              # LR scheduler configurations
```

## Training

### 1. Tokenizer Training

Train a tokenizer to learn efficient visual representations. Defaults use DINOv3-B QFormer encoder with MMDiT-L decoder and discrete diffusion. Trained using 8 H100 GPUS.

```bash
# Single GPU training with defaults
uv run train_tokenizer.py

# With 256x256 images
uv run train_tokenizer.py \
  ++dataset.image_size=256 \
  ++dataset.dinov2_image_size=256
```

### 2. World Model Training

Train the CDiT world model. Defaults use CDiT-B with maksed token modeling + history masking and CompactTok tokenizer. Trained using 4 RTX 6000 ada GPUs.

```bash
# Distributed training (requires tokenizer checkpoint)
uv run bash scripts/train.sh --nproc=4 -- \
  ++tokenizer_path=<TOKENIZER_DIR>

# With larger model
uv run bash scripts/train.sh --nproc=8 -- \
  ++tokenizer_path=<TOKENIZER_DIR> \
  model/generator=cdit_xl
```

## Planning Evaluation

```bash
# Run planning evaluation
uv run bash scripts/plan.sh --nproc=1 -- \
  ++exp_dir=<WORLD_MODEL_DIR>

# Multi-GPU with custom checkpoint
uv run bash scripts/plan.sh --nproc=4 -- \
  ++exp_dir=<WORLD_MODEL_DIR> \
  ++ckp="latest"
```

## Acknowledgements

This codebase builds on the following repositories:
- [NWM](https://github.com/facebookresearch/nwm/)
- [MAGE](https://github.com/LTH14/mage)
- [SEED-Voken](https://github.com/TencentARC/SEED-Voken)
- [FlexTok](https://github.com/apple/ml-flextok)

## Citation

```bibtex
@inproceedings{kim2026planning,
  title={Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model},
  author={Kim, Dongwon and Seo, Gawon and Lee, Jinsung and Cho, Minsu and Kwak, Suha},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```