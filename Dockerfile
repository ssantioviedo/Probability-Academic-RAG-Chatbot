# Use python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build tools for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy local files to the container
COPY . .

# Install python dependencies
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# Run Streamlit
# --server.port=7860 is critical for Hugging Face
# --server.address=0.0.0.0 allows external access
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
