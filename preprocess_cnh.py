#!/usr/bin/env python3
"""
Pré-processamento avançado para melhorar OCR em CNH
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image

def preprocess_image(image_path):
    """Aplica múltiplas técnicas de pré-processamento"""

    # Carregar imagem
    img = cv2.imread(image_path)
    print(f"Imagem original: {img.shape}")

    # Mostrar informações da imagem
    print(f"Tipo: {img.dtype}, Min: {img.min()}, Max: {img.max()}")

    results = {}

    # Técnica 1: Converter para grayscale
    print("\n1. Grayscale básico...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text1 = pytesseract.image_to_string(gray, lang='por')
    results['grayscale'] = len(text1)
    print(f"   Extraído: {len(text1)} chars")
    if len(text1) > 100:
        print(f"   Texto: {text1[:200]}")

    # Técnica 2: Binarização com Otsu
    print("\n2. Binarização Otsu...")
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text2 = pytesseract.image_to_string(binary, lang='por')
    results['otsu'] = len(text2)
    print(f"   Extraído: {len(text2)} chars")
    if len(text2) > 100:
        print(f"   Texto: {text2[:200]}")

    # Técnica 3: Adaptive threshold
    print("\n3. Adaptive Threshold...")
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    text3 = pytesseract.image_to_string(adaptive, lang='por')
    results['adaptive'] = len(text3)
    print(f"   Extraído: {len(text3)} chars")
    if len(text3) > 100:
        print(f"   Texto: {text3[:200]}")

    # Técnica 4: Aumentar contraste (CLAHE)
    print("\n4. CLAHE (contrast enhancement)...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    text4 = pytesseract.image_to_string(enhanced, lang='por')
    results['clahe'] = len(text4)
    print(f"   Extraído: {len(text4)} chars")
    if len(text4) > 100:
        print(f"   Texto: {text4[:200]}")

    # Técnica 5: Denoising + Sharpening
    print("\n5. Denoise + Sharpen...")
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    text5 = pytesseract.image_to_string(sharpened, lang='por')
    results['denoise_sharpen'] = len(text5)
    print(f"   Extraído: {len(text5)} chars")
    if len(text5) > 100:
        print(f"   Texto: {text5[:200]}")

    # Técnica 6: Redimensionar (melhorar DPI)
    print("\n6. Resize (2x)...")
    height, width = gray.shape
    resized = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    text6 = pytesseract.image_to_string(resized, lang='por')
    results['resize_2x'] = len(text6)
    print(f"   Extraído: {len(text6)} chars")
    if len(text6) > 100:
        print(f"   Texto: {text6[:200]}")

    # Técnica 7: Tesseract com config específico
    print("\n7. Tesseract com config otimizado...")
    custom_config = r'--oem 3 --psm 6'
    text7 = pytesseract.image_to_string(gray, lang='por', config=custom_config)
    results['custom_config'] = len(text7)
    print(f"   Extraído: {len(text7)} chars")
    if len(text7) > 100:
        print(f"   Texto: {text7[:200]}")

    # Mostrar melhor técnica
    best_tech = max(results.items(), key=lambda x: x[1])
    print(f"\n✓ MELHOR TÉCNICA: {best_tech[0]} com {best_tech[1]} caracteres")

    # Salvar imagens processadas para análise visual
    cv2.imwrite('/tmp/cnh_gray.jpg', gray)
    cv2.imwrite('/tmp/cnh_binary.jpg', binary)
    cv2.imwrite('/tmp/cnh_adaptive.jpg', adaptive)
    cv2.imwrite('/tmp/cnh_clahe.jpg', enhanced)
    print("\nImagens processadas salvas em /tmp/cnh_*.jpg")

    return results, best_tech

print("="*80)
print("ANÁLISE DE PRÉ-PROCESSAMENTO - CNH")
print("="*80)

results, best = preprocess_image('/tmp/test_img1.jpg')

print("\n" + "="*80)
print("RESUMO DOS RESULTADOS")
print("="*80)
for tech, chars in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{tech:20s}: {chars:4d} caracteres")
