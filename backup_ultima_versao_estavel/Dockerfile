FROM python:3.10-slim

WORKDIR /app

# Configurar pip para maior timeout e retry
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=5

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    build-essential \
    g++ \
    tesseract-ocr \
    tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip, setuptools e wheel primeiro
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar dependências em camadas para evitar timeout
# Camada 1: Dependências base
RUN python3 -m pip install --no-cache-dir \
    numpy==1.23.5 \
    Pillow==10.0.0

# Camada 2: PaddlePaddle
RUN python3 -m pip install --no-cache-dir --timeout=600 \
    paddlepaddle==2.6.2

# Camada 3: Todas as dependências do PaddleOCR 2.7 (baseado no requirements.txt oficial)
RUN python3 -m pip install --no-cache-dir --timeout=600 \
    shapely==2.0.6 \
    scipy==1.10.1 \
    scikit-image \
    imgaug \
    pyclipper \
    lmdb \
    tqdm \
    visualdl \
    rapidfuzz \
    opencv-python-headless==4.6.0.66 \
    opencv-contrib-python==4.6.0.66 \
    cython \
    lxml \
    premailer \
    openpyxl \
    attrdict \
    pyyaml \
    Polygon3 \
    lanms-neo \
    python-Levenshtein

# Camada 4: Instalar PaddleOCR 2.7.0.0 sem dependências (já instaladas acima)
RUN python3 -m pip install --no-cache-dir --timeout=600 --no-deps \
    paddleocr==2.7.0.0

# Camada 5: FastAPI e dependências da API
RUN python3 -m pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    gunicorn \
    python-multipart \
    requests

# Camada 6: PyMuPDF para suporte a PDF (versão compatível com PaddleOCR 2.7)
RUN python3 -m pip install --no-cache-dir \
    PyMuPDF==1.20.2

# Camada 7: Tesseract OCR fallback para documentos de baixa qualidade
RUN python3 -m pip install --no-cache-dir \
    pytesseract

# Copia o servidor
COPY server.py /app/server.py

# Expõe porta 8000
EXPOSE 8000

# Healthcheck para o SaladCloud - usando curl para maior compatibilidade
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Inicia a API - 0.0.0.0 para aceitar conexões externas
# --workers 4: 4 processos worker para paralelismo (4 × 3 pool = 12 instâncias OCR)
# --preload: Carrega app antes de fazer fork para evitar race condition no download de modelos
# Capacidade esperada: ~240 req/min (vs ~23 req/min anterior)
CMD ["gunicorn", "server:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--preload"]
