FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch first to avoid triton/GPU packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY app.py .
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 7860
EXPOSE 8000

ENV GROQ_API_KEY=""
ENV PINECONE_API_KEY=""
ENV POSTGRES_USER=""
ENV SQLPWD=""
ENV OPENAI_API_KEY=""

CMD ["./start.sh"]
