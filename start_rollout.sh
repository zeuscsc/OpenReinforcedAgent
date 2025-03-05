docker run -it --name rollout-1 --rm --gpus '"device=1"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            --network host \
            -v $(pwd):/workspace \
            grpo:dev \
            python /workspace/grpo_rollout.py \
            --dataset /workspace/Qwen2.5-7B-Instruct-qlora/temp_dataset_0_device_1 \
            --device cuda \
            --model /workspace/vllm-models/vllm-7 \
            --num_rollouts 4 \
            --vllm_port 8001