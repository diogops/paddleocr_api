#!/usr/bin/env python3
"""
Análise detalhada da CNH com múltiplas técnicas de OCR
"""

from PIL import Image
import cv2
import numpy as np
import pytesseract

def analyze_image(image_path, image_name):
    """Analisa imagem com diferentes técnicas"""
    print(f"\n{'='*80}")
    print(f"ANÁLISE DA IMAGEM: {image_name}")
    print(f"{'='*80}\n")

    # Carregar imagem
    img_pil = Image.open(image_path)
    img_cv = cv2.imread(image_path)

    print(f"Dimensões: {img_pil.size} (width x height)")
    print(f"Modo: {img_pil.mode}")

    # OCR direto com Tesseract (sem rotação)
    print("\n--- OCR DIRETO (0°) ---")
    text_direct = pytesseract.image_to_string(img_pil, lang='por')
    print(f"Caracteres extraídos: {len(text_direct)}")
    print(f"Texto:\n{text_direct[:500]}\n")

    # Testar diferentes rotações
    rotations = [0, 90, 180, 270]
    results = []

    print("\n--- TESTANDO ROTAÇÕES ---")
    for angle in rotations:
        if angle == 0:
            rotated = img_pil
        else:
            rotated = img_pil.rotate(-angle, expand=True)

        text = pytesseract.image_to_string(rotated, lang='por')
        char_count = len(text.strip())
        results.append({
            'angle': angle,
            'text': text,
            'chars': char_count
        })
        print(f"  Rotação {angle:3}°: {char_count:4} caracteres")

    # Melhor resultado
    best = max(results, key=lambda x: x['chars'])
    print(f"\n✓ MELHOR ROTAÇÃO: {best['angle']}° com {best['chars']} caracteres")

    # Mostrar texto completo da melhor rotação
    print(f"\n--- TEXTO COMPLETO (Rotação {best['angle']}°) ---")
    print(best['text'])

    # Analisar dados estruturados
    print(f"\n--- DADOS IDENTIFICADOS ---")
    text_upper = best['text'].upper()

    # CPF
    import re
    cpf_patterns = [r'\d{3}\.\d{3}\.\d{3}-\d{2}', r'\d{11}']
    for pattern in cpf_patterns:
        matches = re.findall(pattern, best['text'])
        if matches:
            print(f"CPF encontrado: {matches}")

    # Datas
    date_pattern = r'\d{2}[/-]\d{2}[/-]\d{4}'
    dates = re.findall(date_pattern, best['text'])
    if dates:
        print(f"Datas encontradas: {dates}")

    # RG/Registro
    rg_pattern = r'\d{7,10}'
    rgs = re.findall(rg_pattern, best['text'])
    if rgs:
        print(f"Números identificados: {rgs[:5]}")

    return best

# Analisar ambas as imagens
result1 = analyze_image('/tmp/test_img1.jpg', 'Imagem 1')
result2 = analyze_image('/tmp/test_img2.jpg', 'Imagem 2')

# Comparar
print(f"\n{'='*80}")
print("COMPARAÇÃO")
print(f"{'='*80}")
print(f"Imagem 1: {result1['chars']} caracteres (rotação {result1['angle']}°)")
print(f"Imagem 2: {result2['chars']} caracteres (rotação {result2['angle']}°)")

if result1['text'] == result2['text']:
    print("\n⚠️ AS IMAGENS SÃO IDÊNTICAS (mesmo conteúdo OCR)")
