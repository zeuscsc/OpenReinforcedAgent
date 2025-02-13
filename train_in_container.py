import os
import torch
import wandb
from transformers import AutoModelForCausalLM
from peft import PeftModel
from trainer import RLTrainer, RLTrainingArguments
from datasets import load_from_disk
import logging
import argparse
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train_with_reference(
    base_model_path: str,
    current_model_path: str,
    dataset_path: str,
    output_dir: str,
    rollout_results_path: str,
    training_args_path: str
):
    """Run training with reference model in container environment"""
    try:
        # 1. Load dataset
        dataset = load_from_disk(dataset_path)
        logging.info("Dataset loaded successfully")
        
        # 2. Load reference model on GPU 0 and training model on GPU 1
        # Reference model (base model without LoRA)
        reference_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}  # Force reference model to GPU 0
        )
        
        # Training model (base model with LoRA)
        training_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:1"}  # Force training model to GPU 1
        )
        training_model = PeftModel.from_pretrained(training_model, current_model_path)
        
        # 3. Load training arguments from JSON
        with open(training_args_path, 'r') as f:
            args_dict = json.load(f)
        
        # 4. Setup training arguments
        training_args = RLTrainingArguments(
            **args_dict,
            device="cuda:1",  # Force training on GPU 1
            reference_model=reference_model,  # Pass reference model to trainer
            reference_device="cuda:0"  # Specify reference model device
        )

        # 5. Load rollout results
        with open(rollout_results_path, 'r') as f:
            rollout_results = json.load(f)

        # 6. Initialize trainer and train
        trainer = RLTrainer(
            model=training_model,
            args=training_args,
            train_dataset=dataset,
            rollout_results=rollout_results
        )
        
        train_results = trainer.train()
        
        # 7. Log metrics
        metrics = train_results.metrics
        wandb.log({
            "loss": metrics.get("train_loss", 0),
            "reward": metrics.get("train_reward", 0)
        })
        
        # 8. Save checkpoint
        trainer.save_model(output_dir)
        
        # 9. Clean up GPU memory
        del training_model
        del reference_model
        del trainer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--current-model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollout-results-path", required=True)
    parser.add_argument("--training-args-path", required=True)
    
    args = parser.parse_args()
    
    train_with_reference(
        base_model_path=args.base_model_path,
        current_model_path=args.current_model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        rollout_results_path=args.rollout_results_path,
        training_args_path=args.training_args_path
    )
