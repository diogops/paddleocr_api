#!/usr/bin/env python3
"""
Teste direto com PaddleOCR para comparar com Tesseract
"""

import sys
sys.path.insert(0, '/opt/paddleocr_api')

from paddleocr import PaddleOCR
import cv2

print("Inicializando PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=False, lang='pt', show_log=False)

print("\nProcessando CNH com PaddleOCR...")
result = ocr.ocr('/tmp/test_img1.jpg', cls=False)

# Extrair todo o texto
all_text = []
if result and result[0]:
    print(f"\nTotal de linhas detectadas: {len(result[0])}\n")

    for idx, line in enumerate(result[0], 1):
        if line and len(line) > 1:
            bbox = line[0]  # Coordenadas da bounding box
            text_info = line[1]  # (text, confidence)
            text = text_info[0]
            confidence = text_info[1]

            all_text.append(text)
            print(f"{idx:3d}. [{confidence:.3f}] {text}")

combined_text = ' '.join(all_text)

print("\n" + "="*80)
print("TEXTO COMBINADO")
print("="*80)
print(combined_text)
print(f"\nTotal: {len(combined_text)} caracteres")
print(f"Linhas: {len(all_text)}")
