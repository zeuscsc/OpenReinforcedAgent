# NeMo-Megatron GRPO with LoRA

This directory contains the implementation of Generative Reinforcement Learning from Preference Optimization (GRPO) using NVIDIA's NeMo 2.0 framework with LoRA (Low-Rank Adaptation) for efficient fine-tuning of the Qwen2.5-7B-Instruct model.

## Overview

This implementation leverages NVIDIA's NeMo 2.0 framework to perform GRPO training with LoRA. GRPO is a reinforcement learning approach that uses human preferences to optimize language models. Unlike traditional RLHF methods that require a separate reward model, GRPO directly optimizes the policy using preference data.

The key components of this implementation include:

- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning using Low-Rank Adaptation
- **GRPO Training**: Implementation of the GRPO algorithm for preference-based optimization
- **NeMo 2.0 Integration**: Leveraging NVIDIA's NeMo framework for distributed training and optimization

## Directory Structure

```
nemo-megatron/
├── configs/
│   └── qwen2.5_7b_lora_grpo.yaml  # Configuration for GRPO training with LoRA
├── scripts/
│   ├── train_grpo_lora.py         # Main training script
│   └── evaluate_model.py          # Evaluation script for the fine-tuned model
├── src/
│   ├── grpo_trainer.py            # GRPO trainer implementation
│   └── data_utils.py              # Data utilities for preference datasets
└── README.md                      # This file
```

## Requirements

- NVIDIA GPU with CUDA support
- NeMo 2.0 framework
- PyTorch 2.0+
- Transformers library
- NVIDIA TensorRT-LLM (optional, for inference acceleration)

## Installation

1. Install the NeMo framework:

```bash
pip install nemo-toolkit[all]
```

2. Install additional dependencies:

```bash
pip install transformers datasets
```

## Data Preparation

The GRPO training requires preference data in the following format:

```json
{
  "prompt": "User prompt text",
  "chosen": "Preferred response",
  "rejected": "Less preferred response"
}
```

Each line in the JSONL file should contain one such JSON object.

You can create dummy data for testing using:

```bash
python -m src.data_utils
```

This will create sample preference data in the `/home/xentropy/OpenReinforcedAgent/data/` directory.

## Training

To train the model with GRPO and LoRA:

```bash
python scripts/train_grpo_lora.py
```

This will use the default configuration from `configs/qwen2.5_7b_lora_grpo.yaml`. You can override configuration parameters using Hydra's syntax:

```bash
python scripts/train_grpo_lora.py trainer.devices=2 model.micro_batch_size=2
```

## Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate_model.py --model-path /path/to/trained_model.nemo
```

You can also provide custom prompts for evaluation:

```bash
python scripts/evaluate_model.py --model-path /path/to/trained_model.nemo --prompts-file /path/to/prompts.json --output-file results.json
```

## Configuration

The main configuration file `configs/qwen2.5_7b_lora_grpo.yaml` contains settings for:

- Model architecture and training parameters
- LoRA configuration (rank, alpha, dropout, target modules)
- GRPO hyperparameters (KL coefficient, clip range)
- Optimization settings (learning rate, weight decay, scheduler)

## Key Features

1. **LoRA Implementation**: Efficient fine-tuning by only updating a small set of adapter parameters
2. **GRPO Algorithm**: Direct optimization from preference data without a separate reward model
3. **NeMo 2.0 Integration**: Leveraging NVIDIA's framework for scalable training

## References

- [NeMo Framework Documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/index.html)
- [Parameter Efficient Fine-Tuning (PEFT) in NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/peft.html)
- [Model Alignment by RLHF in NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/modelalignment/rlhf.html)
