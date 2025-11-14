#!/usr/bin/env python3
"""
Script para testar detecção de rotação em imagens
Testa múltiplas rotações e mostra qual extrai mais texto
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
import tempfile
import os

def test_single_rotation(ocr, img_array, angle):
    """Testa uma única rotação"""
    try:
        # Rotacionar imagem
        if angle == 90:
            rotated = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img_array, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated = img_array

        # Salvar temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Fazer OCR
        results = ocr.ocr(tmp_path, cls=False)

        # Extrair texto
        all_text = []
        if results and results[0]:
            for line in results[0]:
                if line and len(line) > 1:
                    text = line[1][0]
                    all_text.append(text)

        extracted_text = ' '.join(all_text)

        # Limpar arquivo temporário
        os.unlink(tmp_path)

        return {
            'angle': angle,
            'text': extracted_text,
            'char_count': len(extracted_text.strip()),
            'line_count': len(all_text)
        }

    except Exception as e:
        print(f"Erro ao processar rotação {angle}°: {e}")
        return {
            'angle': angle,
            'text': '',
            'char_count': 0,
            'line_count': 0
        }

def main():
    print("=" * 80)
    print("TESTE DE DETECÇÃO DE ROTAÇÃO")
    print("=" * 80)

    # Inicializar PaddleOCR
    print("\n[1/4] Inicializando PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=False, lang='pt')

    # Carregar imagens
    print("\n[2/4] Carregando imagens de teste...")
    img1_path = "/tmp/img1.jpg"
    img2_path = "/tmp/img2.jpg"

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("ERRO: Não foi possível carregar as imagens!")
        return

    print(f"  - Imagem 1: {img1.shape[1]}x{img1.shape[0]}px")
    print(f"  - Imagem 2: {img2.shape[1]}x{img2.shape[0]}px")

    # Testar rotações
    rotations = [0, 90, 180, 270]

    for img_num, img in enumerate([img1, img2], 1):
        print(f"\n[{img_num+2}/4] Testando rotações para IMAGEM {img_num}:")
        print("-" * 80)

        results = []
        for angle in rotations:
            print(f"  Testando rotação {angle:3}°...", end=" ")
            result = test_single_rotation(ocr, img, angle)
            results.append(result)
            print(f"{result['char_count']:4} chars, {result['line_count']:2} linhas")

        # Encontrar melhor rotação
        best = max(results, key=lambda x: x['char_count'])
        print(f"\n  ✓ MELHOR ROTAÇÃO: {best['angle']}° ({best['char_count']} chars)")
        print(f"  ✓ Texto extraído (primeiros 300 chars):")
        print(f"    {best['text'][:300]}")

        # Mostrar texto completo da melhor rotação
        print(f"\n  ✓ TEXTO COMPLETO ({best['char_count']} chars):")
        print(f"    {best['text']}")
        print()

if __name__ == "__main__":
    main()
