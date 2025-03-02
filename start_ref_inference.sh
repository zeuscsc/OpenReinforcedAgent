docker run -it --name ref-inference-0 --rm --gpus '"device=0"' --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 python grpo_ref_logprob.py \
                 --model-path /workspace/Qwen2.5-7B-Instruct-bnb-4bit \
                 --dataset-path /workspace/Qwen2.5-7B-Instruct-qlora-grpo/temp_dataset_0_device_0_tokenized