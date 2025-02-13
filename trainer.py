import torch
import transformers
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, List
from packaging import version
from dataclasses import dataclass
from llm_rollout import run_llm_rollout, calculate_rewards, get_mrr, get_answer_similarity, get_format_reward


class GRPOTrainer(Trainer):
    def __init__(
        self,
        model,
        beta=0.1,  # KL penalty coefficient
        max_length=None,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.beta = beta
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self._metrics = {"completion_length": [], "kl": []}
        self._last_loaded_step = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Except the following keys in the inputs:
        - input_ids: the rollout input_ids
        - ref_logprobs: the logprobs computed from the reference model
        - completion_ids: a mask indicating which tokens are completions
        - advantage: the advantage computed from the rollout
        """
        if return_outputs:
            raise ValueError("The RLTrainer does not support returning outputs")

        # Get model device
        device = next(model.parameters()).device
        
        # Get inputs
        input_ids = inputs.get("input_ids")
        completion_mask = inputs.get("completion_ids")  # Mask indicating which tokens are completions
        advantages = inputs.get("advantage")
        ref_logprobs = inputs.get("ref_logprobs")
        
        if any(x is None for x in [input_ids, completion_mask, advantages, ref_logprobs]):
            raise ValueError("Missing required inputs: input_ids, completion_ids (mask), advantage, or ref_logprobs")
            
        if self.pad_token_id is None:
            raise ValueError("pad_token_id must be set for padding")

        # Move to device
        input_ids = input_ids.to(device)
        completion_mask = completion_mask.to(device)
        advantages = advantages.to(device)
        ref_logprobs = ref_logprobs.to(device)
        
        # Get batch size and sequence lengths
        batch_size = input_ids.shape[0]
        seq_lengths = (input_ids != self.pad_token_id).sum(dim=1)
        max_seq_len = seq_lengths.max().item()
        
        # If max_length is set, ensure we don't exceed it
        if self.max_length is not None:
            max_seq_len = min(max_seq_len, self.max_length)
        
        # Pad or truncate input_ids and completion_mask
        if input_ids.shape[1] != max_seq_len:
            # Create new tensors with proper size
            new_input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, 
                                     dtype=input_ids.dtype, device=device)
            new_completion_mask = torch.zeros((batch_size, max_seq_len), 
                                           dtype=completion_mask.dtype, device=device)
            
            # Copy data, handling both padding and truncation
            for i in range(batch_size):
                seq_len = min(seq_lengths[i], max_seq_len)
                new_input_ids[i, :seq_len] = input_ids[i, :seq_len]
                new_completion_mask[i, :seq_len] = completion_mask[i, :seq_len]
            
            input_ids = new_input_ids
            completion_mask = new_completion_mask

        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token_id).float()

        # Compute logprobs for current model
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]  # exclude last token prediction
        log_probs = logits.log_softmax(dim=-1)
        current_logprobs = torch.gather(
            log_probs, 
            dim=-1, 
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out non-completion tokens
        current_logprobs = current_logprobs * completion_mask[:, 1:]
        ref_logprobs = ref_logprobs * completion_mask[:, 1:]

        # Compute KL divergence (only for completion tokens)
        kl = torch.exp(ref_logprobs - current_logprobs) - (ref_logprobs - current_logprobs) - 1
        kl = kl * completion_mask[:, 1:].float()  # Ensure KL is only computed for completion tokens

        # Compute policy loss with KL penalty
        policy_loss = -(current_logprobs * advantages.unsqueeze(1) - self.beta * kl)
        
        # Average over completion tokens only (excluding padding)
        num_completion_tokens = completion_mask[:, 1:].sum()
        loss = policy_loss.sum() / (num_completion_tokens + 1e-8)

        # Update metrics
        self._metrics["completion_length"].append(completion_mask.sum().item())
        self._metrics["kl"].append((kl.sum() / num_completion_tokens).item())

        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics during training."""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs.update(metrics)
        
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
            
        self._metrics = {key: [] for key in self._metrics}