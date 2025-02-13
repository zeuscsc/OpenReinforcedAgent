import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from dataclasses import dataclass
import logging
from typing import List, Dict
import numpy as np

logging.basicConfig(level=logging.INFO)

@dataclass
class InferenceInput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_mask: torch.Tensor

def load_inputs(input_path: str) -> InferenceInput:
    """Load preprocessed inputs from JSON"""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    return InferenceInput(
        input_ids=torch.tensor(data['input_ids']),
        attention_mask=torch.tensor(data['attention_mask']),
        completion_mask=torch.tensor(data['completion_mask'])
    )

def save_outputs(logprobs: torch.Tensor, output_path: str):
    """Save logprobs to file"""
    with open(output_path, 'w') as f:
        json.dump({
            'ref_logprobs': logprobs.cpu().numpy().tolist()
        }, f)

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    # Get command line arguments
    input_path = os.environ.get('INPUT_PATH')
    output_path = os.environ.get('OUTPUT_PATH')
    model_path = os.environ.get('MODEL_PATH')
    
    if not all([input_path, output_path, model_path]):
        raise ValueError("Missing required environment variables")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load inputs
    inputs = load_inputs(input_path)
    
    # Prepare model and inputs for distributed inference
    model = accelerator.prepare(model)
    input_ids = accelerator.prepare(inputs.input_ids)
    attention_mask = accelerator.prepare(inputs.attention_mask)
    completion_mask = accelerator.prepare(inputs.completion_mask)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]  # exclude last token prediction
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply completion mask
        token_log_probs = token_log_probs * completion_mask[:, 1:]
    
    # Gather results from all processes
    token_log_probs = accelerator.gather(token_log_probs)
    
    # Save results (only on main process)
    if accelerator.is_main_process:
        save_outputs(token_log_probs, output_path)

if __name__ == "__main__":
    main()
