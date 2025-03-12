#!/usr/bin/env python3

"""
Data utilities for GRPO training with NeMo 2.0
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import GPTDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

class PreferenceDataset(Dataset):
    """
    Dataset for preference data (chosen vs rejected responses)
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 2048,
    ):
        """
        Initialize preference dataset
        
        Args:
            file_path: Path to the JSONL file containing preference data
            tokenizer: Tokenizer to use for encoding
            max_seq_length: Maximum sequence length
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load data
        self.data = self._load_data()
        logging.info(f"Loaded {len(self.data)} preference pairs from {file_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Ensure the required fields are present
                if 'prompt' in item and 'chosen' in item and 'rejected' in item:
                    data.append(item)
                else:
                    logging.warning(f"Skipping item due to missing fields: {item}")
        return data
    
    def _prepare_sample(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        """
        Prepare a sample by tokenizing and formatting
        
        Args:
            prompt: The input prompt
            response: The response (chosen or rejected)
            
        Returns:
            Dictionary containing input_ids and attention_mask
        """
        # Format as instruction following format for Qwen2.5
        formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        
        # Tokenize
        encodings = self.tokenizer(
            formatted_text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract tensors
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def __len__(self) -> int:
        """Return the number of preference pairs"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a preference pair
        
        Args:
            idx: Index of the preference pair
            
        Returns:
            Dictionary containing chosen and rejected samples
        """
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # Prepare chosen and rejected samples
        chosen_sample = self._prepare_sample(prompt, chosen)
        rejected_sample = self._prepare_sample(prompt, rejected)
        
        return {
            "chosen_input_ids": chosen_sample["input_ids"],
            "chosen_attention_mask": chosen_sample["attention_mask"],
            "rejected_input_ids": rejected_sample["input_ids"],
            "rejected_attention_mask": rejected_sample["attention_mask"]
        }

def load_datasets(cfg) -> Tuple[DataLoader, DataLoader]:
    """
    Load train and validation datasets
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Initialize tokenizer
    tokenizer = get_nmt_tokenizer(
        library=cfg.model.tokenizer.library,
        model_name=cfg.model.tokenizer.model_name,
        tokenizer_model=cfg.model.tokenizer.type,
    )
    
    # Create datasets
    train_dataset = PreferenceDataset(
        file_path=cfg.model.data.train_ds.file_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.data.train_ds.max_seq_length
    )
    
    val_dataset = PreferenceDataset(
        file_path=cfg.model.data.validation_ds.file_path,
        tokenizer=tokenizer,
        max_seq_length=cfg.model.data.validation_ds.max_seq_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.model.data.train_ds.micro_batch_size,
        shuffle=cfg.model.data.train_ds.shuffle,
        num_workers=cfg.model.data.train_ds.num_workers,
        pin_memory=cfg.model.data.train_ds.pin_memory
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.model.data.validation_ds.micro_batch_size,
        shuffle=cfg.model.data.validation_ds.shuffle,
        num_workers=cfg.model.data.validation_ds.num_workers,
        pin_memory=cfg.model.data.validation_ds.pin_memory
    )
    
    return train_dataloader, val_dataloader

def create_dummy_preference_data(output_path: str, num_samples: int = 100):
    """
    Create dummy preference data for testing
    
    Args:
        output_path: Path to save the JSONL file
        num_samples: Number of preference pairs to generate
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    for i in range(num_samples):
        item = {
            "prompt": f"This is a test prompt {i}. Please provide a helpful response.",
            "chosen": f"This is a helpful chosen response for prompt {i}. It provides clear and accurate information.",
            "rejected": f"This is a rejected response for prompt {i}. It is not very helpful or contains incorrect information."
        }
        data.append(item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    logging.info(f"Created {num_samples} dummy preference pairs at {output_path}")

if __name__ == "__main__":
    # Create dummy data for testing
    logging.basicConfig(level=logging.INFO)
    os.makedirs("/home/xentropy/OpenReinforcedAgent/data", exist_ok=True)
    create_dummy_preference_data("/home/xentropy/OpenReinforcedAgent/data/train_preferences.jsonl", 1000)
    create_dummy_preference_data("/home/xentropy/OpenReinforcedAgent/data/val_preferences.jsonl", 200)
