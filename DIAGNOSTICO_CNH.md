# Diagnóstico do Problema OCR - CNH JOSE BENEDITO

## Resumo do Problema

A API PaddleOCR está extraindo **240 caracteres** internamente, mas retornando apenas **66 caracteres** na resposta HTTP.

## Evidências dos Logs

### Logs do Docker (Processamento Interno)
```
Testando múltiplas rotações em paralelo (racing)...
[2025/11/09 02:37:29] ppocr DEBUG: dt_boxes num : 2, elapse : 3.7952888011932373
[2025/11/09 02:37:29] ppocr DEBUG: dt_boxes num : 3, elapse : 3.6747093200683594
[2025/11/09 02:37:29] ppocr DEBUG: dt_boxes num : 9, elapse : 4.220811367034912
[2025/11/09 02:37:29] ppocr DEBUG: dt_boxes num : 14, elapse : 4.079524278640747
[2025/11/09 02:37:29] ppocr DEBUG: rec_res num  : 2, elapse : 0.1585400104522705
  Rotação  270°:   21 chars
[2025/11/09 02:37:29] ppocr DEBUG: rec_res num  : 3, elapse : 0.15430331230163574
  Rotação   90°:    0 chars
[2025/11/09 02:37:29] ppocr DEBUG: rec_res num  : 9, elapse : 0.3013322353363037
  Rotação    0°:   66 chars
[2025/11/09 02:37:29] ppocr DEBUG: rec_res num  : 14, elapse : 0.9052877426147461
  Rotação    0°:  240 chars   <-- CORRETO!

Resultado do racing de rotações:
  ✓ MELHOR   0°:  240 caracteres   <-- CORRETO!
     90°:    5 caracteres
    180°:   40 caracteres
    270°:   77 caracteres
Usando rotação 0° (240 caracteres extraídos)
Documento 1: OCR (240 chars)
Documento 1 concluído: total 240 caracteres (nativo + OCR)
DEBUG: Resultados recebidos: 1
DEBUG: Documento 1: 240 chars
DEBUG: Texto final: 240 chars (de 1 documentos)   <-- CORRETO!
```

### Resposta HTTP Recebida
```json
{
  "ocrText": "ARIA DOSANJOS SOUZA Aaal 046518610728 SANTOANTONIODEJESUS.BA BAHIA",
  "total_chars": 66   <-- ERRADO! Deveria ser 240
}
```

## Análise Técnica

### 1. Rotação Utilizada
- ✅ Rotação **0°** foi escolhida corretamente (imagem já estava na orientação correta)
- ✅ Racing identificou corretamente que 0° tinha mais texto (240 chars)

### 2. Detecção de Bounding Boxes
- O PaddleOCR detectou **14 bounding boxes** (linhas de texto)
- Extraiu **240 caracteres** no total
- Logs mostram `rec_res num : 14` confirmando que reconheceu 14 linhas

### 3. Problema Identificado
**Bug na resposta HTTP**: O texto completo é processado internamente (240 chars), mas apenas uma parte (66 chars) é retornada ao cliente.

## Comparação com Tesseract

### Tesseract (com config otimizado)
```
Config: --oem 3 --psm 6
Resultado: 628 caracteres extraídos

Texto parcial:
REPUBLICA FEDERATIVA DO BRASIL
MINISTERIO DA INFRAESTRUTURA
SECRETARIA NACIONAL DE TRANSITO
CARTEIRA NACIONAL DE HABILITAÇÃO
JOSE BENEDITO SOUZA DA HORA
28/02/1952 FEIRA DE SANTANA
MARIA DOS ANJOS SOUZA
```

### PaddleOCR (interno)
```
Resultado: 240 caracteres extraídos (confirmado pelos logs)
Rotação: 0°
Bounding boxes: 14 linhas
```

### PaddleOCR (resposta HTTP)
```
Resultado: 66 caracteres (TRUNCADO!)
Texto: "ARIA DOSANJOS SOUZA Aaal 046518610728 SANTOANTONIODEJESUS.BA BAHIA"
```

## Possíveis Causas do Bug

### Hipótese 1: Racing Condition
O racing de rotações está executando **duas vezes** para rotação 0°:
- Primeira execução: 9 boxes → 66 chars
- Segunda execução: 14 boxes → 240 chars

Possível problema: a resposta pode estar pegando o resultado errado.

### Hipótese 2: Deduplicação de Imagens
```
Imagem duplicada detectada (hash: eae227ba...) - pulando
Deduplicação Hash: 2 imagens → 1 únicas (1 duplicadas removidas)
```

As duas URLs apontam para a **mesma imagem** (hashes idênticos). Isso está correto.

### Hipótese 3: Problema no Processamento Paralelo
O código usa `asyncio.gather()` para processar múltiplos documentos em paralelo. Pode haver race condition onde o resultado errado sobrescreve o correto.

## Dados Esperados da CNH

```
Nome: JOSE BENEDITO SOUZA DA HORA
CPF: 061.918.605-44
Data nascimento: 28/02/1952
Mãe: MARIA DOS ANJOS SOUZA
Local: SANTO ANTONIO DE JESUS, BA
Categoria: D
RG: 07751018 SSP BA
```

## Recomendações

### 1. URGENTE: Investigar Bug na Resposta HTTP
- Verificar código em `server.py:1064-1095` (função `process_ocr_parallel`)
- Verificar se há race condition onde resultado parcial sobrescreve resultado completo
- Adicionar logs para rastrear qual resultado está sendo retornado

### 2. Melhorar Detecção de Bounding Boxes
- PaddleOCR está detectando menos boxes (14) que Tesseract consegue processar
- Considerar ajustar parâmetros do detector:
  - `det_db_thresh` (threshold de detecção)
  - `det_db_box_thresh` (threshold de confiança da box)
  - `det_limit_side_len` (tamanho limite da imagem)

### 3. Adicionar Tesseract como Fallback
- Quando PaddleOCR extrair < 100 caracteres, tentar Tesseract com config otimizado
- Config recomendado: `--oem 3 --psm 6`
- Tesseract está extraindo 2.6x mais texto (628 vs 240 chars)

### 4. Pré-processamento de Imagem
- Considerar aplicar grayscale antes do OCR
- Técnicas que funcionaram:
  - Grayscale básico: +188 chars (Tesseract)
  - Config otimizado: +628 chars (Tesseract)
  - Denoise + Sharpen: +276 chars (Tesseract)

## Conclusão

**Problema principal**: Bug na resposta HTTP que trunca o texto de 240 para 66 caracteres.

**Próximos passos**:
1. Debugar código de resposta em `process_ocr_parallel`
2. Adicionar logs detalhados
3. Considerar implementar Tesseract fallback
4. Testar com outras CNHs para confirmar se é problema sistemático
