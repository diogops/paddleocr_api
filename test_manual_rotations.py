#!/usr/bin/env python3
"""
Testa manualmente todas as rotações no documento 2 para debug
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import requests
from io import BytesIO

# Inicializar PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='pt',
    show_log=False,
    enable_mkldnn=True,
    cpu_threads=4,
    rec_batch_num=6,
    det_limit_side_len=960
)

# Baixar imagem do documento 2
print("Baixando documento 2...")
url = "https://incontadig.s3.amazonaws.com/2025/10/28/aedd4ce9-49ba-4bf2-a915-1394e9a238a7/usgwl5wixck5ab0zcsjdd_1761652972718.jpeg"
response = requests.get(url)
img_bytes = np.frombuffer(response.content, dtype=np.uint8)
img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

print(f"Imagem carregada: {img.shape[1]}x{img.shape[0]}px\n")

# Testar todas as rotações
rotations = {
    0: "0° (original)",
    90: "90° (sentido anti-horário)",
    180: "180° (invertido)",
    270: "270° (sentido horário / 90° horário)"
}

results = {}

for angle, desc in rotations.items():
    print(f"{'='*80}")
    print(f"Testando rotação: {desc}")
    print('='*80)

    # Rotacionar imagem
    if angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        rotated = img.copy()

    print(f"Tamanho após rotação: {rotated.shape[1]}x{rotated.shape[0]}px")

    # Fazer OCR
    try:
        result = ocr.ocr(rotated, cls=False)

        if result and result[0]:
            texts = [line[1][0] for line in result[0]]
            full_text = ' '.join(texts)

            print(f"Caixas detectadas: {len(result[0])}")
            print(f"Total de caracteres: {len(full_text)}")
            print(f"\nTexto extraído:")
            print("-" * 80)
            print(full_text)
            print("-" * 80)

            # Verificar palavras-chave
            palavras_chave = ["ROSA", "MARIA", "CUNHA", "MACIEIRA", "MARIALVA", "FILIACAO", "NOME"]
            encontradas = [p for p in palavras_chave if p.upper() in full_text.upper()]

            print(f"\nPalavras-chave encontradas ({len(encontradas)}): {', '.join(encontradas) if encontradas else 'NENHUMA'}")

            results[angle] = {
                'text': full_text,
                'boxes': len(result[0]),
                'chars': len(full_text),
                'keywords': len(encontradas)
            }
        else:
            print("Nenhum texto detectado")
            results[angle] = {'text': '', 'boxes': 0, 'chars': 0, 'keywords': 0}
    except Exception as e:
        print(f"Erro: {e}")
        results[angle] = {'text': '', 'boxes': 0, 'chars': 0, 'keywords': 0}

    print()

# Resumo
print(f"{'='*80}")
print("RESUMO COMPARATIVO")
print('='*80)
print(f"{'Rotação':<25} {'Boxes':<10} {'Chars':<10} {'Keywords':<10}")
print('-'*80)
for angle, desc in rotations.items():
    r = results[angle]
    print(f"{desc:<25} {r['boxes']:<10} {r['chars']:<10} {r['keywords']:<10}")

# Melhor rotação
best = max(results.items(), key=lambda x: (x[1]['keywords'], x[1]['chars']))
print('='*80)
print(f"MELHOR ROTAÇÃO: {rotations[best[0]]} ({best[1]['keywords']} palavras-chave, {best[1]['chars']} chars)")
print('='*80)
