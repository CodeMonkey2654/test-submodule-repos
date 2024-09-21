# Use the official Python image as the base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /src

# Copy application code
COPY . /src

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip cache purge  # Clean up pip cache to reduce image size

# Expose the port FastAPI is running on
EXPOSE 8000

# Define the default command to run the app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]