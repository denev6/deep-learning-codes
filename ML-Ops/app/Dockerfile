# From docker-hub: python:3.10-slim
FROM python:3.10-slim

# Copy current(.) to "/app" in docker image
COPY . /app

# Set working dir to "/app"
WORKDIR /app

# Run commands during build process
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Port: 5000
EXPOSE 5000

# Default command
CMD ["python", "app.py"]