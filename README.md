# GRPO on agentic LLM

## AWS Ubuntu Guide
Make sure you know the Docker Engine version and then modfiy the Docker file:
nvcr.io/nvidia/pytorch:xx.xx-py3
Where:
    xx.xx is the container version. For example, 28.01.



### CUDA Docker Container Support
Following the link:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
To test it, run:
```shell
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### Start Docker Container
```shell
docker build -t orfa:latest .
docker compose run -d
# docker exec -it openreinforcedagent-app-1 /bin/bash
docker run --gpus all -it orfa:latest /bin/bash
```

### Once connected to container via CLI
```shell
python data_prep.py
```

# How to use
1. download Qwen2.5-7B-Instruct
2. create QLoRA version of the model with create_lora_model.py
3. Pull and build docker images
4. run data_prep.py to prepare the dataset
5. run grpo_train.py to train the model

# Docker images
1. vllm/vllm-openai:v0.7.2
2. build docker image with Dockerfile, tag it with `docker build grpo:dev .`