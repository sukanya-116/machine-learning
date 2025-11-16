FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY credit_fraud_detect_model.json .
COPY scaler.pkl .
COPY app.py .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]