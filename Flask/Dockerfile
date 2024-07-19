# syntax=docker/dockerfile:1

# Specify the Python version as an argument
ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim as base

# Install required system libraries
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Keep Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Create a non-privileged user with a valid home directory
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Install Python dependencies
# NOTE: The --mount option requires Docker BuildKit. If you're not using BuildKit, remove it.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --no-cache-dir -r requirements.txt

# Create necessary directories and set permissions
RUN mkdir -p /app/uploads /app/.cache && \
    chown -R appuser:appuser /app/uploads /app/.cache

# Copy the source code into the container
COPY . .

# Switch to the non-privileged user
USER appuser

# Expose the port that the application listens on
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
