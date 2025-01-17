# From docker-hub: python:3.10-slim
FROM python:3.10-slim

# Copy files to "/app" in docker image
COPY main.py /app/
COPY static /app/static/
COPY templates /app/templates/

# Set working dir to "/app"
WORKDIR /app

# Run commands during build process
RUN pip install --upgrade pip
RUN pip install \
    fastapi==0.112.2 \
    uvicorn==0.34.0 \
    python-multipart==0.0.9 \
    numpy==1.26.3 \
    pillow==10.2.0
# Torch without CUDA (lightweight)
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Port: 8000
EXPOSE 8000

# Default command
# uvicorn main:app --host=0.0.0.0 --port=8000
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
