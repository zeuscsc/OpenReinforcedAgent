import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk
import logging
import argparse
import os

def run_inference(model_path, dataset_path, start_idx, end_idx):
    """Run inference on a specific portion of the dataset"""
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=f"cuda",
        quantization_config=quantization_config
    )
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    # Create batch indices
    batch_size = 2
    
    # Select only the portion of the dataset for this device
    dataset = dataset.select(range(start_idx, end_idx))
    
    # Create batch indices
    batch_indices = [list(range(i, min(i+batch_size, len(dataset)))) for i in range(0, len(dataset), batch_size)]
    
    all_logprobs = {}
    
    with torch.no_grad():
        for batch_idx, indices in enumerate(batch_indices):
            _dataset = dataset.select(indices)
            _input_ids = torch.tensor(_dataset["input_ids"], dtype=torch.long).to(f"cuda:{device_id}")
            _attention_mask = torch.tensor(_dataset["attention_mask"], dtype=torch.long).to(f"cuda:{device_id}")
            
            # Skip empty tensors
            if _input_ids.shape[0] == 0 or (len(_input_ids.shape) > 0 and _input_ids.shape[0] > 0 and _input_ids.shape[1] == 0):
                print(f"Skipping empty tensor batch {batch_idx}")
                continue
            
            # Ensure tensors have the right dimensions (batch_size, seq_len)
            if len(_input_ids.shape) == 1:
                print(f"Reshaping 1D tensor to 2D for batch {batch_idx}")
                _input_ids = _input_ids.unsqueeze(0)
                _attention_mask = _attention_mask.unsqueeze(0)
            
            # Ensure we have at least one valid example
            if _input_ids.shape[0] == 0:
                print(f"Empty batch after reshaping, skipping batch {batch_idx}")
                continue
            
            outputs = model(
                input_ids=_input_ids,
                attention_mask=_attention_mask
            )
            logits = outputs.logits
            logprobs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()
            
            # Store logprobs with global indices
            for i, idx in enumerate(indices):
                all_logprobs[start_idx + idx] = logprobs[i]
            
            # Free memory
            del _input_ids, _attention_mask, logits, logprobs
            torch.cuda.empty_cache()
    
    # Save results for this portion
    output_path = os.path.join(dataset_path, f"logprobs_{device_id}.pt")
    torch.save(all_logprobs, output_path)
    print(f"Saved logprobs for device {device_id} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--start-idx", type=int, required=True)
    parser.add_argument("--end-idx", type=int, required=True)
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        device_id=args.device_id,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )
