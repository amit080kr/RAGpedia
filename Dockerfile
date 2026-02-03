FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for some pip packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Verify installation during build (will fail build if import fails)
RUN python -c "import sentence_transformers; print('sentence_transformers installed successfully')"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
