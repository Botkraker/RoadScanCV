# Hugging Face Space (Docker, free CPU): Label Studio + the tn model server in one container.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# two environments: Label Studio and Ultralytics pin different library versions
RUN python -m venv /opt/ls && /opt/ls/bin/pip install --no-cache-dir label-studio==1.23.0 psycopg2-binary
RUN python -m venv /opt/ml \
 && /opt/ml/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && /opt/ml/bin/pip install --no-cache-dir ultralytics==8.4.155 pyyaml

WORKDIR /app
COPY . /app
ENV LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/app/data/tn \
    LABEL_STUDIO_BASE_DATA_DIR=/tmp/ls \
    LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true \
    SESSION_COOKIE_SECURE=1 \
    YOLO_CONFIG_DIR=/tmp/yolo
EXPOSE 7860
CMD ["sh", "start.sh"]
