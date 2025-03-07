# GRPO on agentic LLM

# How to use
1. download Qwen2.5-7B-Instruct
2. create QLoRA version of the model with create_lora_model.py
3. Pull and build docker images
4. run data_prep.py to prepare the dataset
5. run grpo_train.py to train the model

# Docker images
1. vllm/vllm-openai:v0.7.2
2. build docker image with Dockerfile, tag it with `docker build grpo:dev .`