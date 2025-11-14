# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PaddleOCR REST API server built with FastAPI. The project provides multiple endpoints for optical character recognition (OCR) with Portuguese language support, including:
- File upload endpoint
- Base64 image processing (no download needed)
- URL-based batch processing with field extraction

**Performance**: Optimized with MKL-DNN, multi-threading, and batch processing for 2-3x faster inference compared to default settings.

## Architecture

The codebase consists of:
- **server.py**: FastAPI application with multiple OCR endpoints
- **Dockerfile**: Container setup with specific dependency versions to avoid numpy/OpenCV conflicts

### Endpoints

1. **`POST /ocr`**: Upload file for OCR (returns lines with scores)
2. **`POST /ocr/base64`**: Process base64 image directly (NEW - no download needed)
3. **`POST /ocr/extract`**: Batch process URLs with field extraction (CPF, RG, names, dates)
4. **`GET /health`**: Health check

### Performance Optimizations

PaddleOCR is initialized with CPU optimizations:
- **enable_mkldnn=True**: Intel MKL-DNN acceleration
- **cpu_threads=4**: Multi-threading for CPU inference
- **rec_batch_num=6**: Batch recognition processing
- **limit_side_len=960**: Limit image size for faster processing
- **use_angle_cls=False**: Disabled for speed (enable if rotated images are common)

Image optimization: Large images (>2000px) are automatically resized to reduce processing time.

## Development Commands

### Running Locally
```bash
# Install dependencies (match Dockerfile versions to avoid conflicts)
pip install numpy==1.23.5 opencv-python-headless==4.6.0.66 paddlepaddle==2.6.2 shapely==2.0.6 scipy==1.10.1
pip install --no-deps paddleocr==2.7.0.0
pip install fastapi uvicorn[standard] python-multipart

# Run the server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Build and Run
```bash
# Build the Docker image
docker build -t paddleocr-api .

# If build is stuck at "unpacking" (common on Docker Desktop for Windows):
# Cancel with Ctrl+C and clean up Docker cache first
docker system prune -a --volumes
docker build -t paddleocr-api .

# Run the container
docker run -p 8000:8000 paddleocr-api
```

### Testing the API

#### Test /ocr endpoint (file upload)
```bash
curl -X POST "http://localhost:8000/ocr" -F "file=@path/to/image.jpg"

# Response:
# {
#   "lines": [
#     {"text": "detected text 1", "score": 0.98},
#     {"text": "detected text 2", "score": 0.95}
#   ]
# }
```

#### Test /ocr/base64 endpoint (base64 image)
```bash
# Using the test script
python3 test_base64.py documento.jpg
python3 test_base64.py documento.jpg true  # with field extraction

# Or with curl (example with small base64)
curl -X POST "http://localhost:8000/ocr/base64" \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_string_here", "extract_fields": false}'
```

#### Test /ocr/extract endpoint (URLs with field extraction)
```bash
curl -X POST "http://localhost:8000/ocr/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
  }'

# Response includes extracted fields (CPF, RG, names, dates, etc.)
```

See **README_BASE64.md** for detailed examples and usage.

## Important Notes

### Dependency Conflicts
The Dockerfile explicitly uses:
- `numpy==1.23.5`
- `opencv-python-headless==4.6.0.66`
- `paddlepaddle==2.6.2`
- `shapely==2.0.6`
- `scipy==1.10.1`
- `paddleocr==2.7.0.0`

These specific versions are pinned to avoid compatibility issues between numpy, OpenCV, and PaddlePaddle. The build process:
1. Installs numpy, opencv-python-headless, paddlepaddle, shapely, and scipy with exact versions
2. Installs paddleocr with `--no-deps` to prevent it from pulling incompatible dependency versions
3. Installs FastAPI and related packages separately

**Common Issues:**
- **"numpy.core.multiarray failed to import"**: This indicates numpy ABI version mismatch. Ensure numpy==1.23.5 is installed.
- **System dependencies required**: `libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1` (OpenCV headless typically doesn't need libgl1)
- **Docker build stuck at "unpacking"**: This is common on Docker Desktop for Windows with large images. The PaddleOCR dependencies create a ~2GB+ image. Either wait (can take 10+ minutes) or cancel and run `docker system prune -a --volumes` to clean cache first.

### Language Configuration
PaddleOCR is configured for Portuguese (`lang='pt'`). To change the language, modify the initialization in server.py:8. Common language codes: `'pt'` (Portuguese), `'en'` (English), `'latin'`, etc.
