#!/bin/bash
echo "Starting FastAPI backend..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Waiting for backend to be ready..."
sleep 5

echo "Starting Streamlit frontend..."
streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false

# If Streamlit exits, also stop backend
kill $BACKEND_PID
