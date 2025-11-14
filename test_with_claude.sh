#!/bin/bash
#
# Script para testar a API PaddleOCR com Claude Fallback
#
# USO:
#   1. Obter API key em: https://console.anthropic.com/
#   2. Executar: ./test_with_claude.sh sk-ant-api03-YOUR_KEY_HERE
#

set -e

if [ -z "$1" ]; then
    echo "❌ Erro: API key não fornecida"
    echo ""
    echo "Uso: $0 <ANTHROPIC_API_KEY>"
    echo ""
    echo "Exemplo:"
    echo "  $0 sk-ant-api03-xxxxxxxxxxxxxxxxxxxxx"
    echo ""
    echo "Para obter a API key:"
    echo "  1. Acesse: https://console.anthropic.com/"
    echo "  2. Vá em API Keys → Create Key"
    echo "  3. Copie a chave (formato: sk-ant-...)"
    exit 1
fi

ANTHROPIC_API_KEY="$1"

echo "======================================================================"
echo "TESTE: PaddleOCR com Claude API Fallback"
echo "======================================================================"
echo ""

# Parar container antigo se existir
echo "1. Limpando containers antigos..."
docker stop paddleocr-claude 2>/dev/null || true
docker rm paddleocr-claude 2>/dev/null || true

# Iniciar container com API key
echo "2. Iniciando container com Claude API habilitado..."
docker run -d -p 8000:8000 \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  --name paddleocr-claude \
  paddleocr-api:claude-fallback

echo "3. Aguardando inicialização (2 minutos)..."
sleep 120

# Testar health
echo "4. Testando health check..."
curl -s http://localhost:8000/health | jq .

# Ver logs de inicialização
echo ""
echo "5. Verificando se Claude API foi habilitada..."
docker logs paddleocr-claude 2>&1 | grep -i claude | head -5

echo ""
echo "======================================================================"
echo "TESTE 1: Documento RG (deve usar Tesseract ou Claude)"
echo "======================================================================"
cat > /tmp/test_rg.json << 'EOF'
{
  "urls": [
    "https://incontadig.s3.amazonaws.com/2025/10/21/72f97c05-d810-4571-8a07-9ee607823edd/9zuatrx23db9vis5k1myz6_1761062697274.jpg"
  ]
}
EOF

echo "Executando OCR..."
time curl -s -X POST "http://localhost:8000/ocr/extract" \
  -H "Content-Type: application/json" \
  -d @/tmp/test_rg.json | jq -r '.status, .total_chars, (.ocrText | .[0:200])'

echo ""
echo "Ver logs do processamento:"
docker logs paddleocr-claude --tail 50 | grep -A 10 "Testando múltiplas"

echo ""
echo "======================================================================"
echo "TESTE COMPLETO!"
echo "======================================================================"
echo ""
echo "Para ver logs completos:"
echo "  docker logs paddleocr-claude"
echo ""
echo "Para testar outras imagens:"
echo '  curl -X POST "http://localhost:8000/ocr/extract" \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"urls": ["https://example.com/sua-imagem.jpg"]}'"'"
echo ""
echo "Para parar o container:"
echo "  docker stop paddleocr-claude"
echo ""
