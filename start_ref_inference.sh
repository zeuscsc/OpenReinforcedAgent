docker run -it --name ref-inference --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                grpo:dev \
                accelerate launch --num_processes=2 grpo_ref_logprob.py \
                --model-path /workspace/Qwen2.5-7B-Instruct-bnb-4bit \
                --dataset-path /workspace/Qwen2.5-7B-Instruct-qlora-grpo/temp_dataset_0