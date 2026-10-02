FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch keeps the image small; use a CUDA base image + default torch wheel for GPU serving.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt
COPY . .
RUN pip install --no-deps .
ENV MODEL_PATH=/app/artifacts/fngft_real.pt
EXPOSE 8080
CMD ["uvicorn", "fngft.api:app", "--host", "0.0.0.0", "--port", "8080"]
