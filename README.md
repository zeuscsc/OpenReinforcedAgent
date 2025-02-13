# GRPO on agentic LLM

# How to use
1. download Llama-3.2-3B-Instruct
2. create LoRA version of the model with create_lora_model.py
3. run data_prep.py to prepare the dataset
4. run train_loop.py to train the model

# Docker images
1. vllm/vllm-openai:v0.7.2
2. build docker image with Dockerfile, tag it with `docker build grpo:dev .`