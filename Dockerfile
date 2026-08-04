# Base Python Slim Image
FROM python:3.9-slim

# System dependencies (like git or compiler if needed by xgboost/catboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working Directory
WORKDIR /app

# Copy Requirements
COPY requirements.txt .

# Install Packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY src/ ./src/
COPY templates/ ./templates/
COPY static/ ./static/
COPY setup.py .
COPY main.py .
COPY app.py .
COPY Telco-Customer-Churn.csv .

# Install local package
RUN pip install -e .

# Expose port
EXPOSE 5000

# Train model first, then run Flask web app
CMD ["sh", "-c", "python main.py && python app.py"]
