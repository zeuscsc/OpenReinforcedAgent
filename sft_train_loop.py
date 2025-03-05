import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from peft import AutoPeftModelForCausalLM, prepare_model_for_kbit_training
import torch.nn.functional as F
from torch import nn
import bitsandbytes as bnb
from typing import List, Dict
from accelerate import PartialState

model_name = "/workspace/Qwen2.5-7B-Instruct-qlora"
dataset = load_from_disk("/workspace/sft_dataset")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16 # needed for FSDP / DS3
)

model = AutoPeftModelForCausalLM.from_pretrained(
    model_name,
    is_trainable=True,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    use_cache=True,
    device_map={'':PartialState().local_process_index}
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
#model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8, mean_resizing=False)

#model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,
    warmup_steps = 5,
    num_train_epochs = 3, # Set this for 1 full training run.
    learning_rate = 1e-4,
    optim = "adamw_8bit",
    remove_unused_columns=False,
    bf16 = True,
    logging_steps = 1,
    weight_decay = 1e-4,
    lr_scheduler_type = "cosine",
    output_dir = model_name + "-sft",
    report_to = "none", # Use this for WandB etc
    use_liger_kernel=True,
    #gradient_checkpointing=True,
    #gradient_checkpointing_kwargs={"use_reentrant": True},
    label_names=["labels"],
    max_grad_norm=1.0,
)

def data_collator(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(samples) == 0:
        return {}
    
    # Get max length in this batch
    # max_length = max(len(sample["input_ids"]) for sample in samples)
    max_length = 2048 + 512
    
    # Initialize tensors
    batch_size = len(samples)
    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    labels = torch.ones((batch_size, max_length), dtype=torch.long)*-100
    
    # Left pad or truncate all sequences
    for i, sample in enumerate(samples):
        seq_len = min(len(sample["input_ids"]), max_length)
        input_ids[i, -seq_len:] = torch.tensor(sample["input_ids"])[-seq_len:]
        attention_mask[i, -seq_len:] = 1
        labels[i, -seq_len:] = torch.tensor(sample["labels"])[-seq_len:]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

decay_parameters = get_parameter_names(model, [nn.LayerNorm])
decay_parameters = [name for name in decay_parameters if "bias" not in name]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if n in decay_parameters],
        "weight_decay": training_args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
        "weight_decay": 0.0,
    },
]

optimizer_kwargs = {
    "betas": (training_args.adam_beta1, training_args.adam_beta2),
    "eps": training_args.adam_epsilon,
}
optimizer_kwargs["lr"] = training_args.learning_rate
adam_bnb_optim = bnb.optim.Adam8bit(
    optimizer_grouped_parameters,
    betas=(training_args.adam_beta1, training_args.adam_beta2),
    eps=training_args.adam_epsilon,
    lr=training_args.learning_rate,
)

sft_trainer = Trainer(
    model = model,
    train_dataset = dataset,
    data_collator=data_collator,
    #optimizers=(adam_bnb_optim, None),
    args=training_args
)

trainer_stats = sft_trainer.train()

if sft_trainer.is_fsdp_enabled:
    sft_trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
sft_trainer.save_model()
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")