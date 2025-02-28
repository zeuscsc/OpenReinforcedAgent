from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
import torch.nn.functional as F
from unsloth import is_bfloat16_supported
from typing import List, Dict

dataset = load_from_disk("sft_dataset")

max_seq_length = 16384 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen2.5-7B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

training_args = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    num_train_epochs = 3, # Set this for 1 full training run.
    learning_rate = 1e-5,
    remove_unused_columns=False,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    output_dir = "Qwen2.5-7B-Instruct-lora-sft",
    report_to = "none", # Use this for WandB etc
)

def data_collator(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(samples) == 0:
        return {}
    
    # Get max length in this batch
    max_length = max(len(sample["input_ids"]) for sample in samples)
    
    # Initialize tensors
    batch_size = len(samples)
    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    completion_masks = torch.zeros((batch_size, max_length), dtype=torch.float)
    
    # Left pad all sequences
    for i, sample in enumerate(samples):
        seq_len = len(sample["input_ids"])
        input_ids[i, -seq_len:] = torch.tensor(sample["input_ids"])
        attention_mask[i, -seq_len:] = 1
        completion_masks[i, -seq_len:] = torch.tensor(sample["completion_masks"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_masks": completion_masks,
    }

class SFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model, 
        inputs, 
        return_outputs=False, 
        num_items_in_batch=None
    ):
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits[:, :-1, :]  # shift left to align with labels
        
        # Prepare labels and masks
        labels = inputs["input_ids"][:, 1:]  # shift right to align with logits
        completion_masks = inputs["completion_masks"][:, :-1]  # align with logits
        
        # Compute masked loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        
        # Apply completion mask and normalize
        loss = loss.reshape(logits.shape[0], -1) * completion_masks
        loss = loss.sum() / (completion_masks.sum() + 1e-8)  # add small epsilon to avoid division by zero
        
        return (loss, outputs) if return_outputs else loss

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

sft_trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset= dataset,
    data_collator=data_collator,
    args=training_args
)

trainer_stats = sft_trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")