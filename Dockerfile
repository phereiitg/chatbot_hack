FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages
# This helps prevent potential build issues on Railway
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

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