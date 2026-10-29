# Final-solution image (Rule: organizers mount the test set at /data, collect /result/submission.jsonl).
# Use the base image given in the organizers' Docker guideline (Blackwell GPUs need CUDA >= 12.8).
FROM vllm/vllm-openai:v0.10.1.1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
# Model weights are copied into the image at build time (no downloads at run time):
#   huggingface-cli download Qwen/Qwen3-14B --local-dir models/Qwen3-14B   (before docker build)
COPY models/ /models/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["bash", "/app/predict.sh"]
