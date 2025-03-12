import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)

def compute_log_probs(model, tokenizer, prompt, continuation=None, max_length=100):
    """Compute log probabilities for a given prompt and optional continuation.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        prompt: The input prompt text
        continuation: Optional continuation text to compute probabilities for
        max_length: Maximum length to consider for continuation
        
    Returns:
        A dictionary containing log probabilities and related information
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    
    # If continuation is provided, compute log probs for those specific tokens
    if continuation:
        # Tokenize the continuation
        continuation_tokens = tokenizer(continuation, add_special_tokens=False).input_ids
        # Limit to max_length tokens
        continuation_tokens = continuation_tokens[:max_length]
        
        # Combine prompt and continuation for forward pass
        combined_input_ids = torch.cat([input_ids, torch.tensor([continuation_tokens]).to(model.device)], dim=1)
        
        # Get the logits from the model
        with torch.no_grad():
            outputs = model(combined_input_ids)
            logits = outputs.logits
        
        # Calculate log probabilities for each continuation token
        log_probs = []
        prompt_length = input_ids.shape[1]
        
        for i in range(prompt_length - 1, combined_input_ids.shape[1] - 1):
            next_token_id = combined_input_ids[0, i + 1].item()
            token_logits = logits[0, i, :]
            token_probs = torch.nn.functional.softmax(token_logits, dim=0)
            token_log_prob = torch.log(token_probs[next_token_id]).item()
            log_probs.append(token_log_prob)
        
        # Calculate total log probability
        total_log_prob = sum(log_probs)
        avg_log_prob = total_log_prob / len(log_probs) if log_probs else 0
        
        return {
            "log_probs": log_probs,
            "total_log_prob": total_log_prob,
            "avg_log_prob": avg_log_prob,
            "continuation_tokens": continuation_tokens,
            "continuation_text": continuation
        }
    
    # If no continuation is provided, just return the input processing
    return {
        "input_ids": input_ids,
        "prompt": prompt
    }

def test_models(quantized_model_path, merged_model_path, test_prompts=None, continuations=None):
    """Test if the merged model gives similar log probabilities as the input quantized model."""
    if test_prompts is None:
        test_prompts = [
            "Explain the concept of reinforcement learning in simple terms.",
            "What are the advantages of using LoRA for fine-tuning language models?",
            "Write a short poem about artificial intelligence.",
            "How does quantization affect model performance?",
            "Summarize the key points of transformer architecture."
        ]
    
    # Default continuations if none provided
    if continuations is None:
        continuations = [
            "Reinforcement learning is a type of machine learning where an agent learns",
            "LoRA (Low-Rank Adaptation) offers several advantages for fine-tuning language models:",
            "In silicon minds and digital hearts,\nA new form of wisdom slowly starts.",
            "Quantization reduces the precision of the model's weights, which",
            "The transformer architecture revolutionized NLP with several key innovations:"
        ]
    
    # Ensure we have a continuation for each prompt
    if len(continuations) < len(test_prompts):
        continuations.extend([None] * (len(test_prompts) - len(continuations)))
    
    # Load quantized model
    logging.info(f"Loading quantized model from {quantized_model_path}")
    quantized_model = AutoModelForCausalLM.from_pretrained(
        quantized_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    quantized_tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    
    # Load merged model
    logging.info(f"Loading merged model from {merged_model_path}")
    merged_model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    merged_tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    
    # Compare log probabilities
    logging.info("Comparing model log probabilities...")
    
    for i, (prompt, continuation) in enumerate(zip(test_prompts, continuations)):
        logging.info(f"\nTest prompt {i+1}: {prompt}")
        if continuation:
            logging.info(f"Continuation: {continuation}")
        
        # Compute log probabilities
        quantized_log_probs = compute_log_probs(quantized_model, quantized_tokenizer, prompt, continuation)
        merged_log_probs = compute_log_probs(merged_model, merged_tokenizer, prompt, continuation)
        
        if continuation:
            # Compare log probabilities
            q_avg_log_prob = quantized_log_probs["avg_log_prob"]
            m_avg_log_prob = merged_log_probs["avg_log_prob"]
            
            log_prob_diff = abs(q_avg_log_prob - m_avg_log_prob)
            log_prob_ratio = min(q_avg_log_prob, m_avg_log_prob) / max(q_avg_log_prob, m_avg_log_prob) if max(q_avg_log_prob, m_avg_log_prob) != 0 else 1.0
            
            logging.info(f"Quantized model avg log prob: {q_avg_log_prob:.4f}")
            logging.info(f"Merged model avg log prob: {m_avg_log_prob:.4f}")
            logging.info(f"Absolute difference: {log_prob_diff:.4f}")
            logging.info(f"Ratio (min/max): {log_prob_ratio:.4f}")
            
            # Calculate token-by-token comparison
            if len(quantized_log_probs["log_probs"]) == len(merged_log_probs["log_probs"]):
                token_diffs = [abs(q - m) for q, m in zip(quantized_log_probs["log_probs"], merged_log_probs["log_probs"])]
                max_diff_idx = token_diffs.index(max(token_diffs))
                max_diff_token = continuation[max_diff_idx:max_diff_idx+10] if max_diff_idx < len(continuation) else "<end>"
                
                logging.info(f"Max token-level difference: {token_diffs[max_diff_idx]:.4f} at position {max_diff_idx} ('{max_diff_token}...')")
            
            # Evaluate similarity
            if log_prob_diff < 0.1:
                logging.info("✅ Log probabilities are very similar!")
            elif log_prob_diff < 0.5:
                logging.info("⚠️ Log probabilities show some differences.")
            else:
                logging.info("❌ Log probabilities differ significantly.")
    
    logging.info("\nTest completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test if merged model gives similar log probabilities as input quantized model")
    parser.add_argument("--quantized-model", required=True, help="Path to the original quantized model")
    parser.add_argument("--merged-model", required=True, help="Path to the merged model")
    parser.add_argument("--custom-prompts", nargs="+", help="Optional custom prompts to test with")
    parser.add_argument("--custom-continuations", nargs="+", help="Optional continuations to test with (should match number of prompts)")
    
    args = parser.parse_args()
    
    test_prompts = args.custom_prompts if args.custom_prompts else None
    test_continuations = args.custom_continuations if args.custom_continuations else None
    
    test_models(
        quantized_model_path=args.quantized_model,
        merged_model_path=args.merged_model,
        test_prompts=test_prompts,
        continuations=test_continuations
    )
