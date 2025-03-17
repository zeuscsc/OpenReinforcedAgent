#!/usr/bin/env python3

"""
Script for training Qwen2.5-7B-Instruct with GRPO using LoRA in NeMo 2.0
"""

import os
import sys
import torch
import logging
from omegaconf import OmegaConf

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.peft_config import LoraConfig
from nemo.core.config import hydra_runner
from nemo.utils import logging as nemo_logging
from nemo.utils.exp_manager import exp_manager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.grpo_trainer import GRPOTrainer
from src.data_utils import PreferenceDataset

@hydra_runner(config_path="../configs", config_name="qwen2.5_7b_lora_grpo")
def main(cfg) -> None:
    """
    Main training function for GRPO with LoRA
    
    Args:
        cfg: Configuration from Hydra
    """
    # Initialize logging
    nemo_logging.init_logger()
    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")
    
    # Set up experiment manager
    exp_manager(cfg.exp_manager, explicit_log_dir=cfg.exp_manager.explicit_log_dir)
    
    # Load base model from Hugging Face
    logging.info(f"Loading base model: Qwen/Qwen2.5-7B-Instruct")
    model = MegatronGPTModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        hparams_file=None,
        trainer=None
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=cfg.model.lora_tuning.lora_rank,
        alpha=cfg.model.lora_tuning.lora_alpha,
        dropout=cfg.model.lora_tuning.lora_dropout,
        target_modules=cfg.model.lora_tuning.target_modules
    )
    
    # Apply LoRA to the model
    model.add_adapter(lora_config)
    
    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        cfg=cfg,
    )
    
    # Train the model
    trainer.fit()
    
    # Save the final model
    trainer.save_model()
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    main()
