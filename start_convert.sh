docker run -it --name convert --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 python merge_lora.py \
                 --base-model /workspace/Qwen2.5-7B-Instruct \
                 --lora-model /workspace/Qwen2.5-7B-Instruct-qlora/checkpoint-1 \
                 --output-path /workspace/vllm-models/vllm-1