# Use a slim Python image as the base
FROM python:3.12-slim

# Set environment variables for Python and Flask
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000

# Set the working directory
WORKDIR /app

# Copy only requirements first for dependency installation
COPY requirements.txt /app/

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . /app

# Add a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose the Flask app's default port
EXPOSE 5000

# # Add a health check to verify the app is running
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:5000/ || exit 1

# Command to run the Flask app
CMD ["flask", "run"]