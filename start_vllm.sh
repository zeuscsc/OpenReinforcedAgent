docker run -it --name vllm-server --rm --gpus='"device=0"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace -p 8001:8001 \
                 -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 \
                 vllm/vllm-openai:v0.7.2 \
                 --model /workspace/Qwen2.5-7B-Instruct-bnb-4bit \
                 --tool-call-parser hermes \
                 --enable-auto-tool-choice \
                 --enable-lora \
                 --lora-modules grpo=/workspace/Qwen2.5-7B-Instruct-qlora/checkpoint-1 \
                 --quantization bitsandbytes \
                 --load-format bitsandbytes \
                 --gpu-memory-utilization 0.8 \
                 --port 8001 \
                 --max_model_len 8096