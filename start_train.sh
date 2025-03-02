docker run -it --name training --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 accelerate launch grpo_trainer.py \
                 --checkpoint-path /workspace/Qwen2.5-7B-Instruct-qlora-grpo/checkpoint-1 \
                 --dataset-paths /workspace/Qwen2.5-7B-Instruct-qlora-grpo/temp_dataset_0_device_0_tokenized_ref_logprobs \
