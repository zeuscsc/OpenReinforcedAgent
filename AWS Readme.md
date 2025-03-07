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

### Tmux
```shell
tmux new -s grpo
python grpo_train.py > script.log 2>&1
```
Detach from the tmux session: Ctrl+b followed by d
Reattach to tumx session:
```shell
tmux attach -t grpo
```
To Kill the session
```shell
tmux kill-session -t grpo
```

```shell
docker run \
    --gpus all \
    -p 8000:8000 \
    --name vllm-openai-container \
    -d \
    vllm/vllm-openai:v0.7.2
```