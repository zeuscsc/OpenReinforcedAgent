import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, set_seed, Trainer, TrainerArguments
from utils import create_and_prepare_model, create_datasets


def data_collator(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(samples) == 0:
        return {}
    
    # Get max length in this batch
    max_length = max(len(sample["input_ids"]) for sample in samples)
    
    # Initialize tensors
    batch_size = len(samples)
    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    labels = torch.ones((batch_size, max_length), dtype=torch.long)*-100
    
    # Left pad all sequences
    for i, sample in enumerate(samples):
        seq_len = len(sample["input_ids"])
        input_ids[i, -seq_len:] = torch.tensor(sample["input_ids"])
        attention_mask[i, -seq_len:] = 1
        labels[i, -seq_len:] = torch.tensor(sample["labels"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # bnb
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_args.use_4bit_quantization,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_args.bnb_4bit_compute_dtype,
        bnb_4bit_quant_storage_dtype=model_args.bnb_4bit_quant_storage_dtype,
    )
    # model
    # model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        is_trainable=True,
        quantization_config=bnb_config,
        torch_dtype=bnb_config.bnb_4bit_quant_storage_dtype,
    )

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}

    # dataset
    dataset = load_from_disk('sft_dataset')

    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SFTConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)

