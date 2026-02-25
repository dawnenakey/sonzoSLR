FROM python:3.10-slim

WORKDIR /app

# Install PyTorch CPU (smaller) and dependencies
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastapi uvicorn boto3 pydantic opencv-python-headless numpy

COPY main.py .
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
