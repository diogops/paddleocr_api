# Deploy PaddleOCR API no SaladCloud

## 📋 Pré-requisitos

- Conta no SaladCloud: https://portal.salad.com/
- Imagem Docker publicada: `chacallgyn/paddleocr-api:latest`

## 🚀 Configuração no SaladCloud

### 1. Container Configuration

**Container Image:**
```
chacallgyn/paddleocr-api:latest
```

**Container Gateway (Networking):**
- Enabled: `Yes`
- Port: `5000`
- Protocol: `HTTP`

**Environment Variables:**
```
PORT=5000
```

### 2. Health Check Configuration

**Health Check Type:** `HTTP`

**Health Check Path:** `/health`

**Health Check Port:** `5000`

**Health Check Method:** `GET` ou `POST` (ambos funcionam)

**Health Check Interval:** `30s`

**Health Check Timeout:** `10s`

**Start Period:** `90s` (importante: download de modelos PaddleOCR leva ~60-90s)

**Retries:** `3`

### 3. Resources (Recomendado)

**CPU:**
- Minimum: `2 vCPUs`
- Recommended: `4 vCPUs`

**Memory:**
- Minimum: `4 GB`
- Recommended: `8 GB`

**GPU:** `Not required` (PaddleOCR otimizado para CPU)

**Storage:** `10 GB` (para cache de modelos)

### 4. Replicas

**Minimum Replicas:** `1`

**Maximum Replicas:** `5` (ou conforme sua necessidade)

## 🧪 Teste Local Antes do Deploy

```bash
# 1. Pull da imagem
docker pull chacallgyn/paddleocr-api:latest

# 2. Testar localmente na porta 5000
docker run -p 5000:5000 -e PORT=5000 chacallgyn/paddleocr-api:latest

# 3. Em outro terminal, testar health check
curl http://localhost:5000/health
# Resposta esperada: {"status":"ok","service":"paddleocr-api"}

# 4. Testar health check POST
curl -X POST http://localhost:5000/health
# Resposta esperada: {"status":"ok","service":"paddleocr-api"}

# 5. Testar OCR base64
curl -X POST http://localhost:5000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_string_here",
    "extract_fields": false
  }'
```

## 📝 Endpoints Disponíveis

### Health Check
```
GET  /health → {"status":"ok","service":"paddleocr-api"}
POST /health → {"status":"ok","service":"paddleocr-api"}
```

### OCR Endpoints
```
POST /ocr               → Upload de arquivo
POST /ocr/base64        → Imagem em base64
POST /ocr/extract       → Batch processing de URLs
GET  /                  → Lista de endpoints
```

## ⚙️ Configurações Avançadas

### Variáveis de Ambiente Opcionais

```bash
# Porta do serviço (padrão: 5000)
PORT=5000

# Número de workers Gunicorn (padrão: 4)
# Ajustar conforme CPU disponível
WORKERS=4
```

### Alterar porta (se necessário)

Se precisar usar porta diferente no SaladCloud:

```bash
# Environment Variables no SaladCloud
PORT=8080

# Container Gateway Port
8080
```

## 🔍 Troubleshooting

### Container não inicia

1. **Verificar logs no SaladCloud:**
   - Procure por "Inicializando pool de 4 instâncias PaddleOCR..."
   - Download de modelos pode demorar 60-90 segundos

2. **Verificar memória:**
   - Mínimo necessário: 4 GB
   - Recomendado: 8 GB

3. **Verificar health check:**
   - Start Period deve ser >= 90s (tempo de download de modelos)

### Health check falha

1. **Verificar porta:**
   - Porta configurada: `5000`
   - Environment variable PORT: `5000`
   - Container Gateway Port: `5000`
   - Health Check Port: `5000`

2. **Verificar path:**
   - Health Check Path: `/health` (com barra inicial)

3. **Verificar método:**
   - GET ou POST (ambos funcionam)

### Performance lenta

1. **Aumentar CPU:**
   - De 2 vCPUs para 4 vCPUs

2. **Aumentar memória:**
   - De 4 GB para 8 GB

3. **Verificar cache de modelos:**
   - Modelos são baixados apenas na primeira inicialização
   - Após download, ficam em cache

## 📊 Performance Esperada

**Inicialização:**
- Download de modelos: ~60-90 segundos (apenas primeira vez)
- Startup do servidor: ~5-10 segundos

**OCR Processing:**
- Imagem simples (CNH/RG): ~8-12 segundos
- Processamento inclui racing de 4 rotações (0°, 90°, 180°, 270°)

**Capacidade:**
- ~5-8 requisições concorrentes por réplica
- Escalar horizontalmente conforme necessidade

## 🎯 Checklist de Deploy

- [ ] Imagem: `chacallgyn/paddleocr-api:latest`
- [ ] PORT environment variable: `5000`
- [ ] Container Gateway Port: `5000`
- [ ] Health Check Path: `/health`
- [ ] Health Check Port: `5000`
- [ ] Start Period: `90s` ou mais
- [ ] Memória: mínimo 4 GB
- [ ] CPU: mínimo 2 vCPUs

## 🔗 Links Úteis

- **SaladCloud Portal:** https://portal.salad.com/
- **Docker Hub:** https://hub.docker.com/r/chacallgyn/paddleocr-api
- **Documentação SaladCloud:** https://docs.salad.com/

## 💡 Dicas

1. **Use storage persistente** se possível para cache de modelos
2. **Configure auto-scaling** baseado em CPU/Memory usage
3. **Monitore logs** durante primeiras horas para validar performance
4. **Teste health check** antes de configurar load balancer
