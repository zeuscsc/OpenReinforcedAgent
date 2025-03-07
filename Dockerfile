FROM nvcr.io/nvidia/pytorch:25.02-py3

COPY requirements.txt .

RUN pip install -r requirements.txt --ignore-installed
RUN pip uninstall -y flash-attn
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab