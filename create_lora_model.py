from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
import torch
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_lora_model(
    base_model_path: str,
    output_dir: str,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    bias: str = "none",
    target_modules: list = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
):
    """
    Create a LoRA version of the base model.
    
    Args:
        base_model_path: Path to the base model
        output_dir: Directory to save the LoRA model
        r: LoRA attention dimension
        lora_alpha: Alpha parameter for LoRA scaling
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
        target_modules: List of module names to apply LoRA to
    """
    logging.info(f"Loading base model from {base_model_path}")
    
    # Load base model in bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Define LoRA Config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Save LoRA model and config
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logging.info(f"LoRA model saved to {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    # Create LoRA version of the model
    model, tokenizer = create_lora_model(
        base_model_path="Llama-3.2-3B-Instruct",
        output_dir="Llama-3.2-3B-Instruct-lora",
    )
