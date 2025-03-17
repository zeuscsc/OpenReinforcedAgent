#!/usr/bin/env python3

"""
GRPO Trainer implementation for NeMo 2.0
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.utils import average_losses_across_data_parallel_group
from nemo.core.optim.lr_scheduler import WarmupPolicy
from nemo.utils import logging as nemo_logging
from nemo.utils.model_utils import compute_model_parallel_rank

class GRPOTrainer:
    """
    Trainer class for Generative Reinforcement Learning from Preference Optimization (GRPO)
    
    This trainer implements the GRPO algorithm using NeMo 2.0 and LoRA for efficient fine-tuning.
    """
    
    def __init__(self, model: MegatronGPTModel, cfg):
        """
        Initialize the GRPO trainer
        
        Args:
            model: The base language model to be fine-tuned
            cfg: Configuration object
        """
        self.model = model
        self.cfg = cfg
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize reference model (for KL penalty)
        self.ref_model = None
        if self.cfg.model.grpo.ref_model_path:
            self.ref_model = self._load_reference_model()
        else:
            # Clone the base model as reference
            self.ref_model = self._clone_model()
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Set up optimizer
        self.optimizer = self._setup_optimizer()
        
        # Set up learning rate scheduler
        self.lr_scheduler = self._setup_lr_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        logging.info("GRPO Trainer initialized successfully")
    
    def _load_reference_model(self) -> MegatronGPTModel:
        """Load reference model from checkpoint"""
        logging.info(f"Loading reference model from: {self.cfg.model.grpo.ref_model_path}")
        ref_model = MegatronGPTModel.from_pretrained(
            model_name=self.cfg.model.grpo.ref_model_path,
            hparams_file=None,
            trainer=None
        )
        return ref_model
    
    def _clone_model(self) -> MegatronGPTModel:
        """Clone the current model to use as reference"""
        logging.info("Cloning current model to use as reference")
        # In practice, we'd load the same model again from HF
        ref_model = MegatronGPTModel.from_pretrained(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            hparams_file=None,
            trainer=None
        )
        return ref_model
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Set up the optimizer"""
        # Only optimize the LoRA parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.cfg.model.optim.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.cfg.model.optim.lr,
                betas=self.cfg.model.optim.betas,
                weight_decay=self.cfg.model.optim.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.model.optim.name}")
        
        return optimizer
    
    def _setup_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Set up learning rate scheduler"""
        if self.cfg.model.optim.sched.name == 'CosineAnnealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.trainer.max_steps - self.cfg.model.optim.sched.warmup_steps,
                eta_min=self.cfg.model.optim.sched.min_lr
            )
            # Wrap with warmup scheduler
            scheduler = WarmupPolicy(
                optimizer=self.optimizer,
                warmup_steps=self.cfg.model.optim.sched.warmup_steps,
                scheduler=scheduler
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.cfg.model.optim.sched.name}")
        
        return scheduler
    
    def compute_kl_divergence(self, logprobs: torch.Tensor, ref_logprobs: torch.Tensor, 
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference model
        
        Args:
            logprobs: Log probabilities from policy model
            ref_logprobs: Log probabilities from reference model
            mask: Optional mask for padding tokens
            
        Returns:
            KL divergence loss
        """
        kl = logprobs - ref_logprobs
        if mask is not None:
            kl = kl * mask
            return kl.sum() / mask.sum()
        return kl.mean()
    
    def compute_grpo_loss(self, chosen_logprobs: torch.Tensor, rejected_logprobs: torch.Tensor,
                         chosen_ref_logprobs: torch.Tensor, rejected_ref_logprobs: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute GRPO loss
        
        Args:
            chosen_logprobs: Log probabilities for chosen responses
            rejected_logprobs: Log probabilities for rejected responses
            chosen_ref_logprobs: Reference log probabilities for chosen responses
            rejected_ref_logprobs: Reference log probabilities for rejected responses
            mask: Optional mask for padding tokens
            
        Returns:
            Total loss and dictionary of individual loss components
        """
        # Compute KL divergence
        chosen_kl = self.compute_kl_divergence(chosen_logprobs, chosen_ref_logprobs, mask)
        rejected_kl = self.compute_kl_divergence(rejected_logprobs, rejected_ref_logprobs, mask)
        
        # Compute preference loss (similar to DPO)
        chosen_reward = chosen_logprobs - self.cfg.model.grpo.kl_coef * chosen_kl
        rejected_reward = rejected_logprobs - self.cfg.model.grpo.kl_coef * rejected_kl
        
        # Compute loss (log-sigmoid of the difference between chosen and rejected)
        loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
        
        # Return loss components for logging
        loss_components = {
            'loss': loss,
            'chosen_kl': chosen_kl,
            'rejected_kl': rejected_kl,
            'chosen_reward': chosen_reward.mean(),
            'rejected_reward': rejected_reward.mean(),
        }
        
        return loss, loss_components
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step
        
        Args:
            batch: Dictionary containing the batch data
            
        Returns:
            Dictionary of loss values and metrics
        """
        self.model.train()
        
        # Extract batch data
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        
        # Forward pass for chosen response
        chosen_outputs = self.model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            labels=chosen_input_ids,
            return_dict=True
        )
        chosen_logprobs = -chosen_outputs.loss
        
        # Forward pass for rejected response
        rejected_outputs = self.model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            labels=rejected_input_ids,
            return_dict=True
        )
        rejected_logprobs = -rejected_outputs.loss
        
        # Get reference model logprobs
        with torch.no_grad():
            # Forward pass for chosen response with reference model
            chosen_ref_outputs = self.ref_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                labels=chosen_input_ids,
                return_dict=True
            )
            chosen_ref_logprobs = -chosen_ref_outputs.loss
            
            # Forward pass for rejected response with reference model
            rejected_ref_outputs = self.ref_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                labels=rejected_input_ids,
                return_dict=True
            )
            rejected_ref_logprobs = -rejected_ref_outputs.loss
        
        # Compute GRPO loss
        loss, loss_components = self.compute_grpo_loss(
            chosen_logprobs, rejected_logprobs,
            chosen_ref_logprobs, rejected_ref_logprobs
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update learning rate
        self.lr_scheduler.step()
        
        # Update global step
        self.global_step += 1
        
        # Return loss components and metrics
        return loss_components
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step
        
        Args:
            batch: Dictionary containing the batch data
            
        Returns:
            Dictionary of loss values and metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Extract batch data
            chosen_input_ids = batch['chosen_input_ids'].to(self.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_input_ids = batch['rejected_input_ids'].to(self.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
            
            # Forward pass for chosen response
            chosen_outputs = self.model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                labels=chosen_input_ids,
                return_dict=True
            )
            chosen_logprobs = -chosen_outputs.loss
            
            # Forward pass for rejected response
            rejected_outputs = self.model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                labels=rejected_input_ids,
                return_dict=True
            )
            rejected_logprobs = -rejected_outputs.loss
            
            # Get reference model logprobs
            chosen_ref_outputs = self.ref_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                labels=chosen_input_ids,
                return_dict=True
            )
            chosen_ref_logprobs = -chosen_ref_outputs.loss
            
            rejected_ref_outputs = self.ref_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                labels=rejected_input_ids,
                return_dict=True
            )
            rejected_ref_logprobs = -rejected_ref_outputs.loss
            
            # Compute GRPO loss
            loss, loss_components = self.compute_grpo_loss(
                chosen_logprobs, rejected_logprobs,
                chosen_ref_logprobs, rejected_ref_logprobs
            )
        
        # Return loss components and metrics
        return loss_components
    
    def fit(self):
        """Train the model for the specified number of epochs"""
        logging.info("Starting GRPO training")
        
        # Load datasets
        from src.data_utils import load_datasets
        train_dataloader, val_dataloader = load_datasets(self.cfg)
        
        # Training loop
        for epoch in range(self.cfg.trainer.max_epochs):
            self.epoch = epoch
            logging.info(f"Starting epoch {epoch+1}/{self.cfg.trainer.max_epochs}")
            
            # Training phase
            self.model.train()
            train_losses = []
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Check if we've reached max steps
                if self.global_step >= self.cfg.trainer.max_steps:
                    logging.info(f"Reached maximum steps ({self.cfg.trainer.max_steps}). Stopping training.")
                    break
                
                # Perform training step
                loss_dict = self.train_step(batch)
                train_losses.append(loss_dict['loss'].item())
                
                # Log progress
                if batch_idx % self.cfg.trainer.log_every_n_steps == 0:
                    avg_loss = np.mean(train_losses[-self.cfg.trainer.log_every_n_steps:])
                    logging.info(f"Epoch {epoch+1}, Step {batch_idx}, Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                # Validation
                if batch_idx % self.cfg.trainer.val_check_interval == 0:
                    self._run_validation(val_dataloader)
            
            # End of epoch validation
            val_metrics = self._run_validation(val_dataloader)
            logging.info(f"Epoch {epoch+1} completed. Validation loss: {val_metrics['loss']:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(f"epoch_{epoch+1}")
            
            # Check if we've reached max steps
            if self.global_step >= self.cfg.trainer.max_steps:
                logging.info(f"Reached maximum steps ({self.cfg.trainer.max_steps}). Stopping training.")
                break
        
        logging.info("Training completed")
    
    def _run_validation(self, val_dataloader) -> Dict[str, float]:
        """Run validation and return metrics"""
        self.model.eval()
        val_losses = []
        
        logging.info("Running validation...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= self.cfg.trainer.limit_val_batches:
                    break
                
                loss_dict = self.validation_step(batch)
                val_losses.append(loss_dict['loss'].item())
        
        # Compute average metrics
        avg_loss = np.mean(val_losses)
        metrics = {'loss': avg_loss}
        
        logging.info(f"Validation Loss: {avg_loss:.4f}")
        return metrics
    
    def _save_checkpoint(self, tag: str):
        """Save a checkpoint of the model"""
        checkpoint_dir = os.path.join(self.cfg.exp_manager.exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.cfg.name}_{tag}.nemo")
        logging.info(f"Saving checkpoint to {checkpoint_path}")
        
        self.model.save_to(checkpoint_path)
    
    def save_model(self):
        """Save the final trained model"""
        output_dir = os.path.join(self.cfg.exp_manager.exp_dir, 'final_model')
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f"{self.cfg.name}_final.nemo")
        logging.info(f"Saving final model to {model_path}")
        
        self.model.save_to(model_path)
