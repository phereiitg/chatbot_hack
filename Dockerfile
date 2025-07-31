FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies required for some Python packages
# This helps prevent potential build issues on Railway
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into a target directory
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt


# --- Stage 2: Final Image ---
# This stage creates the final, smaller image for production.
FROM python:3.10-slim

WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
# Railway will automatically use this exposed port.
EXPOSE 8000

# Define environment variable for the port
ENV PORT=8000

# Command to run the application using uvicorn
# This will be executed when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# ==============================================================================
# FILE: project/railway.json
# ==============================================================================
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "uvicorn app:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
