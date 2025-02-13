FROM nvcr.io/nvidia/pytorch:24.03-py3

COPY requirements.txt .

RUN pip install -r requirements.txt