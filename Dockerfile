FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CAUSALRAG_API_HOST=0.0.0.0
ENV CAUSALRAG_API_PORT=8000

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "causalrag.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
