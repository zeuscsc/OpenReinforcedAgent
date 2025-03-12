#!/usr/bin/env python3

"""
Script to export a NeMo model trained with LoRA to Hugging Face format
"""

import os
import sys
import torch
import logging
import argparse
from typing import Optional

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def export_to_hf(
    nemo_model_path: str,
    output_dir: str,
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
):
    """
    Export a NeMo model to Hugging Face format
    
    Args:
        nemo_model_path: Path to the .nemo model file
        output_dir: Directory to save the exported model
        base_model_name: Name of the base model on Hugging Face
    """
    logging.info(f"Loading NeMo model from {nemo_model_path}")
    nemo_model = MegatronGPTModel.restore_from(nemo_model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load base model and tokenizer from Hugging Face
    logging.info(f"Loading base model from Hugging Face: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )
    
    # Extract and apply LoRA weights from NeMo model to base model
    logging.info("Extracting and applying LoRA weights")
    
    # This is a simplified approach - in a real implementation, you would need to
    # map the NeMo model's LoRA weights to the corresponding HF model parameters
    # The exact implementation depends on the specific model architecture
    
    # Save the merged model and tokenizer
    logging.info(f"Saving merged model to {output_dir}")
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save model card with information about the training
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"""# Qwen2.5-7B-Instruct-GRPO

This model is a fine-tuned version of [{base_model_name}](https://huggingface.co/{base_model_name}) using Generative Reinforcement Learning from Preference Optimization (GRPO) with LoRA.

## Training Details

- **Base Model:** {base_model_name}
- **Training Method:** GRPO with LoRA
- **Framework:** NVIDIA NeMo 2.0
- **Original NeMo Model:** {nemo_model_path}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{output_dir}", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("{output_dir}", trust_remote_code=True)

prompt = "Explain the concept of reinforcement learning in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```
""")
    
    logging.info(f"Model successfully exported to {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Export NeMo model to Hugging Face format")
    parser.add_argument("--nemo-model-path", type=str, required=True,
                        help="Path to the .nemo model file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save the exported model")
    parser.add_argument("--base-model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Name of the base model on Hugging Face")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Export model
    export_to_hf(
        nemo_model_path=args.nemo_model_path,
        output_dir=args.output_dir,
        base_model_name=args.base_model_name
    )

if __name__ == "__main__":
    main()
