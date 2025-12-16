# 1. Base Image: Use a lightweight Python version (Slim) to keep image size small
# Enterprise Best Practice: Never use 'latest'. Pin specific versions.
FROM python:3.9-slim

# 2. Set Environment Variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Ensures logs are streamed immediately to the container logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set Working Directory inside the container
WORKDIR /app

# 4. Install Dependencies
# Enterprise Best Practice: Copy requirements FIRST to leverage Docker Layer Caching.
# If you change your code but not your requirements, Docker skips this slow step.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Source Code
# This copies everything from your current folder to /app (respecting .dockerignore)
COPY . .

# 6. Expose the Port
# We need port 8000 open for traffic
EXPOSE 8000

# 7. Command to Run the App
# We bind to 0.0.0.0 so the container is accessible from outside
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]