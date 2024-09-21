# Use a lightweight base image for faster loading
FROM alpine:3.14

# Set working directory
WORKDIR /app

# Install minimal system dependencies for simulation
RUN apk add --no-cache \
    python3 \
    py3-pip \
    build-base \
    python3-dev \
    && rm -rf /var/cache/apk/*

# Copy only necessary application code
COPY ./src/main.py /app/main.py

# Install minimal Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables to avoid geographic area prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Add a simplified entrypoint script
RUN echo '#!/bin/sh' > /entrypoint.sh && \
    echo 'python3 /app/main.py' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Define the entry point
ENTRYPOINT ["/entrypoint.sh"]
