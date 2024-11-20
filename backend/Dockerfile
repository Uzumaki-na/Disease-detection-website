FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set up Kaggle credentials directory
RUN mkdir -p /root/.kaggle

# Create a default kaggle.json (will be overwritten by environment variables)
RUN echo '{"username":"","key":""}' > /root/.kaggle/kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# Copy the rest of the application
COPY . .

# Create directories for datasets and models
RUN mkdir -p datasets/skin_cancer datasets/malaria models

EXPOSE 8000

# Create script to check for models and start the application
RUN echo '#!/bin/bash\n\
# Update Kaggle credentials from environment variables\n\
echo "{\\"username\\":\\"$KAGGLE_USERNAME\\",\\"key\\":\\"$KAGGLE_KEY\\"}" > /root/.kaggle/kaggle.json\n\
chmod 600 /root/.kaggle/kaggle.json\n\
\n\
if [ ! -f /app/models/skin_cancer_model.pth ] || [ ! -f /app/models/malaria_model.pth ]; then\n\
    echo "Models not found. Running training..."\n\
    python3 train_models_torch.py\n\
fi\n\
\n\
exec "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]