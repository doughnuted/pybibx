# Use the official Python 3.11.13 slim image (same ABI as Colab)
FROM python:3.11.13-slim

# Install system deps for notebooks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      curl \
      && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /workspace

# Copy your requirements (or install inline)
RUN pip install --upgrade pip \
    && pip install \
        pandas==2.3.0 \
        numpy==1.26.4 \
        python-dateutil==2.9.0.post0 \
        python-slugify==8.0.4 \
        jupyter

# Expose the notebook port
EXPOSE 8888

# Start JupyterLab on container start
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]