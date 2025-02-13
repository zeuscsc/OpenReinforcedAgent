import os
import torch
import subprocess
import signal
import wandb
from transformers import get_scheduler, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from torch.optim import AdamW
from trainer import GRPOTrainer
from datasets import load_from_disk
import psutil
import logging
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class TrainingManager:
    def __init__(
        self,
        base_model_path: str,
        lora_model_path: str,
        chat_template_path: str,
        dataset_path: str,
        output_dir: str,
        tools: list = None,
        max_steps: int = 1000,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        per_device_train_batch_size: int = 4,
        num_rollouts: int = 32,
        beta: float = 0.1,
    ):
        self.base_model_path = base_model_path
        self.lora_model_path = lora_model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.per_device_train_batch_size = per_device_train_batch_size
        self.num_rollouts = num_rollouts
        self.beta = beta
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        with open(chat_template_path, "r") as f:
            self.tokenizer.chat_template = f.read()

    def kill_docker_container(self, container_name="vllm-server"):
        try:
            cmd = f"docker ps -q --filter name={container_name}"
            container_id = subprocess.check_output(cmd, shell=True).decode().strip()
            if container_id:
                subprocess.run(f"docker stop {container_id}", shell=True)
                logging.info(f"Successfully stopped container {container_id}")
        except Exception as e:
            logging.error(f"Error stopping docker container: {e}")

    def spawn_vllm_server(self, base_model_path, lora_model_path):
        # Kill any existing container
        self.kill_docker_container()
        
        # Start new container with base model and LoRA adapter
        cmd = f"""docker run -d --name vllm-server --rm --gpus='"device=0"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace -p 8000:8000 \
                 vllm/vllm-openai:v0.7.2 \
                 --model /workspace/{base_model_path} \
                 --enable-lora \
                 --lora-modules rl=/workspace/{lora_model_path} \
                 --chat-template /workspace/tool_chat_template_llama3.2_pythonic.jinja \
                 --tool-call-parser pythonic \
                 --enable-auto-tool-choice \
                 --dtype bfloat16
                 --max-context-length 32768"""
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info("Successfully started vLLM server")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start vLLM server: {e}")
            raise

    def spawn_training_container(self, base_model_path, current_model_path, dataset_path, output_dir, rollout_results_path, training_args_path):
        """Start training container with specified parameters"""
        cmd = f"""docker run -d --name grpo:dev --rm --gpus='"device=0,1"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 ora:dev \
                 --base-model-path /workspace/{base_model_path} \
                 --current-model-path /workspace/{current_model_path} \
                 --dataset-path /workspace/{dataset_path} \
                 --output-dir /workspace/{output_dir} \
                 --rollout-results-path /workspace/{rollout_results_path} \
                 --training-args-path /workspace/{training_args_path}"""
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            logging.info("Successfully started training container")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to start training container: {e}")
            raise

    def kill_training_container(self, container_name="training-container"):
        """Kill the training container"""
        try:
            cmd = f"docker ps -q --filter name={container_name}"
            container_id = subprocess.check_output(cmd, shell=True).decode().strip()
            if container_id:
                subprocess.run(f"docker stop {container_id}", shell=True)
                logging.info(f"Successfully stopped container {container_id}")
        except Exception as e:
            logging.error(f"Error stopping training container: {e}")

    def process_rollouts(self, rollout_results):
        """Process rollout results into training inputs"""
        all_input_ids = []
        all_completion_masks = []
        all_rewards = []
        
        for result in rollout_results:
            condition = result["data"]
            completions = result["rollout_results"]
            
            # Calculate rewards for this group of completions
            rewards = calculate_rewards(
                condition=condition,
                rollout_results=completions,
                reward_functions={
                    'mrr': get_mrr,
                    'similarity': get_answer_similarity,
                    'format': get_format_reward
                },
                weight_scheme="std"
            )
            
            # Extract input_ids and create completion masks
            for completion in completions:
                # Apply chat template to get input_ids
                sequence = self.tokenizer.apply_chat_template(completion, tools=tools, tokenize=False)
                sequence = sequence.split('<|eot|>')
                completion_mask = [s.startswith('<|start_header_id|>assistant<|end_header_id|>') for s in sequence]
                sequence = self.tokenizer(sequence, add_special_tokens=False)
                
                completion_mask = [[1]*len(s) if c else [0]*len(s) for s, c in zip(sequence, completion_mask)]
                # Flatten sequence
                sequence = [item for sublist in sequence for item in sublist]
                completion_mask = [item for sublist in completion_mask for item in sublist]
                all_input_ids.append(sequence)
                all_completion_masks.append(completion_mask)
                all_rewards.append(rewards["rewards"])
        
        return {
            "input_ids": all_input_ids,
            "completion_masks": all_completion_masks,
            "rewards": all_rewards
        }

    def run_ref_inference(self, processed_inputs, steps):
        """Run distributed inference with reference model to get logprobs"""
        # Set up output path
        output_path = os.path.join(self.output_dir, f"ref_outputs_{steps}.json")
        
        # Run distributed inference
        cmd = f"""docker run -d --name ref-inference --rm --gpus all --ipc=host \
                 -v $(pwd):/workspace \
                 -e INPUT_PATH=/workspace/{input_path} \
                 -e OUTPUT_PATH=/workspace/{output_path} \
                 -e MODEL_PATH=/workspace/{self.base_model_path} \
                 grpo:dev"""
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            
            # Wait for inference to complete
            while True:
                cmd = "docker ps -q --filter name=ref-inference"
                container_id = subprocess.check_output(cmd, shell=True).decode().strip()
                if not container_id:
                    break
                time.sleep(10)
            
            # Load results
            with open(output_path, 'r') as f:
                ref_outputs = json.load(f)
            
            return torch.tensor(ref_outputs['ref_logprobs'])
            
        except Exception as e:
            logging.error(f"Error during reference inference: {e}")
            raise
        finally:
            # Ensure container is stopped
            subprocess.run("docker stop ref-inference", shell=True)

    def train(self):
        current_model_path = self.lora_model_path
        steps = 0
        # 1. Load dataset
        dataset = load_from_disk(self.dataset_path)
        logging.info("Dataset loaded successfully")

        while steps < self.max_steps:
            logging.info(f"Starting step {steps + 1}/{self.max_steps}")
            
            try:
                # 2. Spawn vLLM server (reference model) on GPU 0
                self.spawn_vllm_server(self.base_model_path, current_model_path)
                logging.info("vLLM server started successfully")
                
                # 3. Run rollouts
                from llm_rollout import run_llm_rollout
                _dataset = Dataset.from_dict(dataset[steps*self.per_device_train_batch_size:(steps+1)*self.per_device_train_batch_size])
                rollout_results = run_llm_rollout(_dataset, device="cuda:1", num_examples=self.num_rollouts)
                
                # 4. Kill vLLM server after rollouts
                self.kill_docker_container()
                logging.info("vLLM server stopped successfully")
                
                # 5. Process rollouts and calculate rewards
                processed_inputs = self.process_rollouts(rollout_results)
                logging.info("Processed rollout results")
                
                # 6. Run distributed inference to get reference logprobs
                ref_logprobs = self.run_ref_inference(processed_inputs, steps)
                logging.info("Computed reference logprobs")
                
                # 7. Save training inputs
                training_inputs = {
                    "input_ids": processed_inputs["input_ids"],
                    "completion_masks": processed_inputs["completion_masks"],
                    "ref_logprobs": ref_logprobs,
                    "advantage": processed_inputs["rewards"] - mean(processed_inputs["rewards"]) / std(processed_inputs["rewards"] + 1e-4)
                }
                
                training_inputs_path = os.path.join(self.output_dir, f"training_inputs_{steps}.json")
                with open(training_inputs_path, 'w') as f:
                    json.dump({k: v.tolist() for k, v in training_inputs.items()}, f)
                
                # 8. Start training container
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{steps}")
                self.spawn_training_container(
                    base_model_path=self.base_model_path,
                    current_model_path=current_model_path,
                    dataset_path=self.dataset_path,
                    output_dir=checkpoint_dir,
                    rollout_results_path=training_inputs_path,
                    training_args_path=os.path.join(self.output_dir, f"training_args_{steps}.json")
                )
                
                # 9. Wait for training to complete and update current model path
                time.sleep(10)  # Give container time to start
                while True:
                    cmd = "docker ps -q --filter name=training-container"
                    container_id = subprocess.check_output(cmd, shell=True).decode().strip()
                    if not container_id:
                        break
                    time.sleep(30)  # Check every 30 seconds
                
                current_model_path = checkpoint_dir
                steps += 1
                
            except Exception as e:
                logging.error(f"Error during training: {e}")
                self.kill_docker_container()
                self.kill_training_container()
                raise
            
        logging.info("Training completed successfully")

if __name__ == "__main__":
    manager = TrainingManager(
        base_model_path="Llama-3.2-3B-Instruct",
        lora_model_path="Llama-3.2-3B-Instruct-lora",
        dataset_path="dataset_curated",
        output_dir="training_outputs",
        max_steps=1000,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_rollouts=32,
    )
    
    manager.train()
