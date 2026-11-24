FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Dependencias del sistema necesarias para Pillow / procesamiento de imagen
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir supabase

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "python3 -m streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true"]