docker run -it --name training --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 accelerate launch --config_file "deepspeed_config.yaml" sft_train_loop.py