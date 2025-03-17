#!/usr/bin/env python3

"""
Script for evaluating NeMo 2.0 models fine-tuned with GRPO and LoRA
"""

import os
import sys
import torch
import logging
import argparse
from typing import List, Optional

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )

def load_model(model_path: str) -> MegatronGPTModel:
    """
    Load a NeMo model from a checkpoint
    
    Args:
        model_path: Path to the .nemo checkpoint file
        
    Returns:
        Loaded model
    """
    logging.info(f"Loading model from {model_path}")
    model = MegatronGPTModel.restore_from(model_path)
    model.eval()
    return model

def generate_response(
    model: MegatronGPTModel,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    num_return_sequences: int = 1
) -> List[str]:
    """
    Generate responses from the model
    
    Args:
        model: The model to generate from
        prompt: Input prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        num_return_sequences: Number of sequences to return
        
    Returns:
        List of generated responses
    """
    # Format as instruction following format for Qwen2.5
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Get tokenizer from model
    tokenizer = model.tokenizer
    
    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode outputs
    responses = []
    for output in outputs:
        # Get only the assistant's response by finding where it starts
        response_start = len(inputs.input_ids[0])
        response_text = tokenizer.decode(output[response_start:], skip_special_tokens=True)
        
        # Clean up the response (remove any trailing system tokens)
        if "<|im_end|>" in response_text:
            response_text = response_text.split("<|im_end|>")[0].strip()
            
        responses.append(response_text)
    
    return responses

def evaluate_model(
    model_path: str,
    test_prompts: Optional[List[str]] = None,
    output_file: Optional[str] = None
):
    """
    Evaluate a model on a set of test prompts
    
    Args:
        model_path: Path to the model checkpoint
        test_prompts: List of test prompts (if None, default prompts will be used)
        output_file: Path to save evaluation results (if None, print to console)
    """
    # Set up default test prompts if none provided
    if test_prompts is None:
        test_prompts = [
            "Explain the concept of reinforcement learning from human feedback.",
            "What are the advantages of using LoRA for fine-tuning language models?",
            "Write a short poem about artificial intelligence.",
            "How does GRPO differ from traditional PPO for language model alignment?",
            "Summarize the key points of transformer architecture."
        ]
    
    # Load model
    model = load_model(model_path)
    
    # Generate responses for each prompt
    results = []
    for i, prompt in enumerate(test_prompts):
        logging.info(f"Processing prompt {i+1}/{len(test_prompts)}")
        responses = generate_response(model, prompt)
        
        result = {
            "prompt": prompt,
            "response": responses[0]
        }
        results.append(result)
        
        # Print to console
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Response: {responses[0]}")
        logging.info("-" * 50)
    
    # Save results to file if specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Evaluation results saved to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate NeMo 2.0 models fine-tuned with GRPO and LoRA")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the .nemo model checkpoint")
    parser.add_argument("--prompts-file", type=str, help="Path to a JSON file containing test prompts")
    parser.add_argument("--output-file", type=str, help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Load test prompts from file if specified
    test_prompts = None
    if args.prompts_file:
        import json
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            test_prompts = json.load(f)
    
    # Evaluate model
    evaluate_model(
        model_path=args.model_path,
        test_prompts=test_prompts,
        output_file=args.output_file
    )

if __name__ == "__main__":
    main()
