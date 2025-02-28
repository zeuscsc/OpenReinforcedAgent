import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import AutoPeftModelForCausalLM
import logging
import json
from awq import AutoAWQForCausalLM


logging.basicConfig(level=logging.INFO)

def merge_lora(
    base_model: str,
    lora_model_path: str,
    output_path: str
):
    """
    Load a LoRA model, dequantize, merge weights and save with uint8 storage
    
    Args:
        lora_model_path: Path to the LoRA adapter weights
        output_path: Where to save the merged model
    """
    logging.info(f"Loading LoRA adapter from {lora_model_path}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_path,
        torch_dtype=torch.bfloat16,
        #torch_dtype=torch.float16, # awq specific
    )
    model.dequantize()

    model.save_pretrained(lora_model_path + '-dequant', save_embedding_layers=False)

    with open(os.path.join(lora_model_path + '-dequant', "adapter_config.json"), "r") as f:
        adapter_config = json.load(f)
        adapter_config["base_model_name_or_path"] = base_model
    
    with open(os.path.join(lora_model_path + '-dequant', "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    del model

    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_model_path + '-dequant',
        torch_dtype=torch.bfloat16
        #torch_dtype=torch.float16, # awq specific
    )
    
    logging.info("Merging LoRA weights...")
    model = model.merge_and_unload()

    model.save_pretrained(output_path, save_embedding_layers=False)
    
    del model

    # BNB 4bit
    # model = AutoModelForCausalLM.from_pretrained(
    #     output_path,
    #     torch_dtype=torch.bfloat16,
    #     trust_remote_code=True,
    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #     )
    # )

    # model.save_pretrained(output_path, save_embedding_layers=False)

    # AWQ
    # model = AutoAWQForCausalLM.from_pretrained(
    #     output_path,
    #     trust_remote_code=True,
    # )

    # tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

    # model.quantize(tokenizer, { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" })
        
    # logging.info(f"Saving merged model to {output_path}")
    
    # model.save_quantized(output_path)

    try:
        AutoTokenizer.from_pretrained(output_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
        tokenizer.save_pretrained(output_path)
    
    logging.info("Merge complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA weights and convert to uint8 storage")
    parser.add_argument("--base-model", required=True, help="Path to base model")
    parser.add_argument("--lora-model", required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--output-path", required=True, help="Where to save the merged model")
    
    args = parser.parse_args()
    
    merge_lora(
        base_model=args.base_model,
        lora_model_path=args.lora_model,
        output_path=args.output_path
    )
