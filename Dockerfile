# 可复现的 CPU 运行环境。
#   构建:   docker build -t face-vision-ai .
#   Hello:  docker run --rm face-vision-ai
#   评测:   docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/weights:/app/weights face-vision-ai python tools/eval_lfw.py
#   界面:   docker run --rm -p 7860:7860 -v ${PWD}/weights:/app/weights face-vision-ai python app/gradio_app.py --host 0.0.0.0
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY . .
ENV PYTHONPATH=/app
CMD ["python", "tools/hello_world.py"]
