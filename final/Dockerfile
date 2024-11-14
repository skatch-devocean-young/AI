# Start from a minimal base image
FROM python:3.10-slim-buster

# Set working directory
WORKDIR /app

# Copy only the requirements first to leverage Docker's cache
COPY requirements.txt /app/requirements.txt

# Install system dependencies and Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Install only what's necessary
    && pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch separately to ensure the correct version is installed
RUN pip install --no-cache-dir torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove unnecessary files and caches
RUN apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/* /root/.cache

# Copy the rest of the application files
COPY . /app

# Expose the port on which the app will run
EXPOSE 5001

# Run the Flask application
CMD ["python", "app.py"]