# Dockerfile for Hugging Face Spaces (Streamlit)
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly set USE_OPENCV flag so we use OpenCV even if MediaPipe installs
ENV USE_OPENCV_HAND_DETECTION=1

# Copy all project files
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
