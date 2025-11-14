# Sistema de Fallback OCR em 3 Níveis

## Visão Geral

Este sistema implementa um **fallback em cascata** para garantir a melhor extração de texto possível de documentos brasileiros (RG, CNH, CPF, etc).

## Arquitetura de Fallback

```
┌─────────────────────────────────────────────────────────────┐
│  NÍVEL 1: PaddleOCR (Rápido, ~10s)                         │
│  - Pré-processamento: CLAHE + Denoise + Binarização        │
│  - Racing paralelo: 4 rotações (0°, 90°, 180°, 270°)       │
│  - Seleção inteligente: prioriza num_boxes                 │
│  - Threshold: < 300 caracteres → Nível 2                   │
└─────────────────────────────────────────────────────────────┘
                          ↓ (se < 300 chars)
┌─────────────────────────────────────────────────────────────┐
│  NÍVEL 2: Tesseract OCR (Fallback, ~5s)                    │
│  - Config otimizado: --oem 3 --psm 6                       │
│  - Racing de rotações com grayscale                        │
│  - Threshold: < 150 caracteres → Nível 3                   │
└─────────────────────────────────────────────────────────────┘
                          ↓ (se < 150 chars)
┌─────────────────────────────────────────────────────────────┐
│  NÍVEL 3: Claude API Multimodal (Último recurso, ~5-10s)   │
│  - Modelo: claude-3-5-sonnet-20241022                      │
│  - Prompt otimizado para documentos brasileiros            │
│  - Extração estruturada: nome, CPF, RG, datas, etc         │
│  - Resultado: texto completo + campos estruturados         │
└─────────────────────────────────────────────────────────────┘
```

## Configuração

### 1. Obter API Key do Anthropic

1. Acesse: https://console.anthropic.com/
2. Crie uma conta ou faça login
3. Vá em **API Keys** → **Create Key**
4. Copie a chave (formato: `sk-ant-...`)

### 2. Configurar Variável de Ambiente

#### Docker (via -e flag)
```bash
docker run -d -p 8000:8000 \
  -e ANTHROPIC_API_KEY="sk-ant-api03-..." \
  --name paddleocr-optimized \
  paddleocr-api:optimized
```

#### Docker Compose
```yaml
services:
  paddleocr:
    image: paddleocr-api:optimized
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=sk-ant-api03-...
```

#### Linux/macOS (shell)
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

#### Windows (PowerShell)
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### 3. Verificar Configuração

```bash
curl http://localhost:8000/health
```

Logs devem mostrar:
```
✓ Claude API habilitada para fallback OCR multimodal
```

## Uso

### API Endpoint

```bash
curl -X POST "http://localhost:8000/ocr/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://exemplo.com/cnh.jpg",
      "https://exemplo.com/rg.jpg"
    ]
  }'
```

### Fluxo de Execução

O sistema tenta automaticamente os 3 níveis:

1. **PaddleOCR** executa primeiro
   - Se extrair ≥ 300 chars → **retorna resultado**
   - Se extrair < 300 chars → tenta Tesseract

2. **Tesseract** executa se PaddleOCR falhou
   - Se extrair mais texto que PaddleOCR:
     - Se extrair ≥ 150 chars → **retorna resultado**
     - Se extrair < 150 chars → tenta Claude
   - Se extrair menos que PaddleOCR:
     - Se PaddleOCR extraiu < 150 chars → tenta Claude
     - Caso contrário → **retorna PaddleOCR**

3. **Claude API** executa apenas quando:
   - Tesseract extraiu < 150 chars **OU**
   - PaddleOCR extraiu < 150 chars e foi melhor que Tesseract

### Logs de Exemplo

```
Imagem redimensionada: 1654x2338 → 1358x1920px
Aplicando pré-processamento avançado na imagem...
✓ Pré-processamento concluído
Testando múltiplas rotações em paralelo (racing)...
  Rotação   0°: 9 boxes, 75 chars
  Rotação  90°: 5 boxes, 21 chars
  Rotação 180°: 5 boxes, 33 chars
  Rotação 270°: 6 boxes, 37 chars
Usando rotação 0° (9 boxes, 75 caracteres extraídos)

⚠️  PaddleOCR extraiu apenas 75 chars. Tentando Tesseract fallback...
Tesseract fallback: testando rotações com config otimizado...
  Tesseract rotação   0°: 123 chars
✓ Tesseract foi melhor: 123 chars vs 75 chars PaddleOCR
⚠️  Tesseract extraiu apenas 123 chars. Tentando Claude API fallback...

🤖 Ativando Claude OCR fallback (multimodal)...
✓ Claude OCR: 487 caracteres extraídos
  Preview: REPÚBLICA FEDERATIVA DO BRASIL CARTEIRA NACIONAL DE HABILITAÇÃO JOSE BENEDITO SOUZA DA HORA CPF: 061.918.605-44 Data Nascimento: 28/02/1952...
✓ Claude foi MELHOR: 487 chars vs 123 chars Tesseract
✓ Usando resultado do Claude (fallback nível 3)
```

## Custos da API Claude

### Modelo: claude-3-5-sonnet-20241022

- **Input**: $3.00 / 1M tokens (~$0.003 por 1k tokens)
- **Output**: $15.00 / 1M tokens (~$0.015 por 1k tokens)

### Custo por Imagem

Estimativa para documentos típicos:
- **Input tokens**: ~1,500 tokens (imagem + prompt)
- **Output tokens**: ~500 tokens (texto extraído)
- **Custo por documento**: ~$0.012 (1.2 centavos de dólar)

### Exemplo de Uso Mensal

| Documentos/mês | Custo estimado |
|----------------|----------------|
| 100            | $1.20          |
| 1,000          | $12.00         |
| 10,000         | $120.00        |
| 100,000        | $1,200.00      |

**Nota**: Claude só é acionado quando PaddleOCR e Tesseract falharem (< 150 chars), o que deve acontecer em < 5% dos casos.

## Desabilitar Claude API

Para desabilitar o fallback do Claude (usar apenas PaddleOCR + Tesseract):

```bash
# Não definir ANTHROPIC_API_KEY ou definir vazio
docker run -d -p 8000:8000 \
  --name paddleocr-optimized \
  paddleocr-api:optimized
```

Logs mostrarão:
```
⚠️  Claude API desabilitada (defina ANTHROPIC_API_KEY para habilitar)
```

## Melhorias Implementadas

### PaddleOCR (Nível 1)
✅ Pool otimizado: 8→4 instâncias
✅ Parâmetros: `det_db_thresh=0.2`, `det_limit_side_len=1920`
✅ Pré-processamento: CLAHE + Denoise + Binarização
✅ Seleção inteligente: prioriza `num_boxes`
✅ Racing paralelo: mantém velocidade ~10s

### Tesseract (Nível 2)
✅ Config otimizado: `--oem 3 --psm 6`
✅ Racing de rotações com grayscale
✅ Threshold ajustado: 200→300 chars para ativação

### Claude API (Nível 3)
✅ Modelo: `claude-3-5-sonnet-20241022` (mais recente)
✅ Prompt otimizado para documentos brasileiros
✅ Extração estruturada: nome, CPF, RG, datas, etc
✅ Threshold: < 150 chars para ativação
✅ Custo controlado: ~$0.012/documento

## Resultados Esperados

### Antes (só PaddleOCR)
- Documentos de boa qualidade: 300-700 chars ✅
- Documentos de baixa qualidade: 50-100 chars ❌
- CNH/RG complexos: 60-200 chars ❌

### Depois (PaddleOCR + Tesseract + Claude)
- Documentos de boa qualidade: 300-700 chars ✅ (PaddleOCR)
- Documentos de baixa qualidade: 200-500 chars ✅ (Tesseract)
- CNH/RG complexos: 400-800 chars ✅ (Claude)

## Troubleshooting

### Claude API não está sendo acionado
```bash
# Verificar se a API key está configurada
docker exec paddleocr-optimized env | grep ANTHROPIC_API_KEY

# Ver logs de inicialização
docker logs paddleocr-optimized | grep Claude
```

### Erro de autenticação
```
❌ Erro no Claude OCR fallback: AuthenticationError: Invalid API key
```
**Solução**: Verificar se a API key está correta e válida

### Timeout na API
```
❌ Erro no Claude OCR fallback: APITimeoutError: Request timed out
```
**Solução**: Verificar conexão com internet. Claude API requer acesso à internet.

### Rate limit atingido
```
❌ Erro no Claude OCR fallback: RateLimitError: Rate limit exceeded
```
**Solução**: Aguardar ou aumentar limite na conta Anthropic

## Monitoramento

### Ver logs em tempo real
```bash
docker logs -f paddleocr-optimized
```

### Estatísticas de uso
Os logs mostram qual nível foi usado:
```bash
docker logs paddleocr-optimized | grep "Usando resultado"
```

Exemplo de output:
```
✓ Usando resultado do Tesseract (fallback nível 2)
✓ Usando resultado do Claude (fallback nível 3)
```

## Segurança

⚠️ **IMPORTANTE**: A API key do Anthropic é sensível!

1. **Não** commitar a chave no código
2. **Não** expor em logs públicos
3. **Use** variáveis de ambiente
4. **Rotacione** a chave regularmente
5. **Configure** limites de rate na console Anthropic

## Suporte

Para problemas ou dúvidas:
- Documentação oficial: https://docs.anthropic.com/
- Console Anthropic: https://console.anthropic.com/
- Status da API: https://status.anthropic.com/
