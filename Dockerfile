FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1

# ───── system deps for OpenCV & LightGBM ────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "UI/app_dl.py", \
     "--server.fileWatcherType", "none", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501"]
