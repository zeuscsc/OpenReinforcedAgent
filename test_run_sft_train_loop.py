from sft_train_loop import SFTTrainingManager
from dotenv import load_dotenv
import os

load_dotenv()

manager = SFTTrainingManager(
    base_model_path="Llama-3.2-3B-Instruct",
    lora_model_path="Llama-3.2-3B-Instruct-lora",
    dataset_path="dataset_curated",
    output_dir="sft_training_outputs",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Replace with actual API key
    max_steps=1000,
    learning_rate=2e-5,
    batch_size=4,
    top_k_samples=32,
    reward_threshold=0.8,
    chat_template_path="tool_chat_template_llama3.2_pythonic.jinja"
)