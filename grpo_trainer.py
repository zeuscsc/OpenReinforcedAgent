import torch
import transformers
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, List
from packaging import version
from dataclasses import dataclass
from grpo_rollout import run_llm_rollout, calculate_rewards, get_mrr, get_answer_similarity, get_format_reward
from torch.nn import functional as F
from utils import pad_and_truncate

class GRPOTrainer(Trainer):
    def __init__(
        self,
        model,
        args,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        beta=0.1,  # KL penalty coefficient
        **kwargs,
    ):
        super().__init__(model, args)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.beta = beta
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
        
        # Get inputs
        input_ids = inputs.get("input_ids").to(self.model.device)
        attention_mask = inputs.get("attention_mask").to(self.model.device)
        labels = inputs.get("labels").to(self.model.device)
        advantages = inputs.get("advantage").to(self.model.device)
        ref_logprobs = inputs.get("ref_logprobs").to(self.model.device)
        
        if any(x is None for x in [input_ids, attention_mask, labels, advantages, ref_logprobs]):
            raise ValueError("Missing required inputs: input_ids, attention_mask, labels, advantage, or ref_logprobs")
        
        # Compute logprobs for current model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)

        completion_mask = torch.gt(labels, -100, dtype=logprobs.dtype)
        # Compute KL divergence (only for completion tokens)
        kl = torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1
        kl = kl  # Ensure KL is only computed for completion tokens

        # Compute policy loss with KL penalty
        policy_loss = -(logprobs * advantages.unsqueeze(1) - self.beta * kl) * completion_mask
        
        # Average over completion tokens only (excluding padding)
        num_completion_tokens = completion_mask.sum()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    dataset = load_from_disk(dataset_path)
    logging.info("Dataset loaded successfully")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16
    )

    # Training model (base model with LoRA)
    model = AutoPeftModelForCausalLM.from_pretrained(
        current_model_path,
        is_trainable=True,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map={'':PartialState().local_process_index}
    )
    
    # 4. Setup training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        bf16=True,
        learning_rate=1e-4,
        num_train_epochs=1,
        weight_decay=0.01,
        output_dir=output_dir,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # 5. Load rollout results
    with open(rollout_results_path, 'r') as f:
        rollout_results = json.load(f)

    # 6. Initialize trainer and train
    trainer = GRPOTrainer(
        model=training_model,
        train_dataset=dataset,
        args=training_args,
        data_collator = lambda x: x,
    )
    
    training_stats = trainer.train(
        resume_from_checkpoint=checkpoint_path
    )
    
    # 8. Save checkpoint
    trainer.save_model(output_dir)