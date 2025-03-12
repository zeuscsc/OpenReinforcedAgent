docker run -it --name rollout-1 --rm --gpus '"device=1"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
            --network host \
            -v $(pwd):/workspace \
            grpo:dev \
            python /workspace/grpo_rollout.py \
            --dataset /workspace/test_rollout \
            --device cuda \
            --model /workspace/vllm-models/vllm-9 \
            --num_rollouts 8 \
            --vllm_port 8000