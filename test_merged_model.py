import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    """Generate a response from the model for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def test_models(quantized_model_path, merged_model_path, test_prompts=None):
    """Test if the merged model gives the same response as the input quantized model."""
    if test_prompts is None:
        test_prompts = [
            "Explain the concept of reinforcement learning in simple terms.",
            "What are the advantages of using LoRA for fine-tuning language models?",
            "Write a short poem about artificial intelligence.",
            "How does quantization affect model performance?",
            "Summarize the key points of transformer architecture."
        ]
    
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
    
    # Compare responses
    logging.info("Comparing model responses...")
    
    for i, prompt in enumerate(test_prompts):
        logging.info(f"\nTest prompt {i+1}: {prompt}")
        
        # Generate responses
        quantized_response = generate_response(quantized_model, quantized_tokenizer, prompt)
        merged_response = generate_response(merged_model, merged_tokenizer, prompt)
        
        # Compare responses
        if quantized_response == merged_response:
            logging.info("✅ Responses match exactly!")
        else:
            logging.info("❌ Responses differ:")
            logging.info(f"Quantized model: {quantized_response}")
            logging.info(f"Merged model: {merged_response}")
            
            # Calculate similarity (simple character-level comparison)
            total_chars = max(len(quantized_response), len(merged_response))
            if total_chars > 0:
                matching_chars = sum(q == m for q, m in zip(
                    quantized_response.ljust(total_chars), 
                    merged_response.ljust(total_chars)
                ))
                similarity = matching_chars / total_chars * 100
                logging.info(f"Response similarity: {similarity:.2f}%")
    
    logging.info("\nTest completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test if merged model gives the same response as input quantized model")
    parser.add_argument("--quantized-model", required=True, help="Path to the original quantized model")
    parser.add_argument("--merged-model", required=True, help="Path to the merged model")
    parser.add_argument("--custom-prompts", nargs="+", help="Optional custom prompts to test with")
    
    args = parser.parse_args()
    
    test_prompts = args.custom_prompts if args.custom_prompts else None
    
    test_models(
        quantized_model_path=args.quantized_model,
        merged_model_path=args.merged_model,
        test_prompts=test_prompts
    )
