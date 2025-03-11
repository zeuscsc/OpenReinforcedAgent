docker run -it --name convert --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
                 -v $(pwd):/workspace \
                 grpo:dev \
                 python data_prep.py