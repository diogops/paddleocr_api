#!/usr/bin/env python3
"""
Script para testar rotações manualmente nas imagens
Usa PIL para rotacionar e depois envia para o servidor
"""

from PIL import Image
import requests
import json
import base64
import io

def rotate_and_test(image_path, angle, image_name):
    """Rotaciona imagem e testa OCR"""
    print(f"\n  Testando rotação {angle:3}° em {image_name}...", end=" ")

    # Carregar e rotacionar imagem
    img = Image.open(image_path)

    # Rotacionar (PIL usa ângulos anti-horários)
    if angle == 90:
        rotated = img.rotate(-90, expand=True)
    elif angle == 180:
        rotated = img.rotate(-180, expand=True)
    elif angle == 270:
        rotated = img.rotate(-270, expand=True)
    else:
        rotated = img

    # Converter para bytes
    img_buffer = io.BytesIO()
    rotated.save(img_buffer, format='JPEG', quality=95)
    img_bytes = img_buffer.getvalue()

    # Converter para base64
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    # Enviar para API
    try:
        response = requests.post(
            'http://localhost:8000/ocr/base64',
            json={'image': img_b64, 'extract_fields': False},
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            lines = data.get('lines', [])
            all_text = ' '.join([line['text'] for line in lines])
            char_count = len(all_text.strip())
            line_count = len(lines)

            print(f"{char_count:4} chars, {line_count:2} linhas")

            return {
                'angle': angle,
                'text': all_text,
                'char_count': char_count,
                'line_count': line_count
            }
        else:
            print(f"ERRO HTTP {response.status_code}")
            return {'angle': angle, 'text': '', 'char_count': 0, 'line_count': 0}

    except Exception as e:
        print(f"ERRO: {e}")
        return {'angle': angle, 'text': '', 'char_count': 0, 'line_count': 0}

def main():
    print("=" * 80)
    print("TESTE MANUAL DE ROTAÇÕES")
    print("=" * 80)

    images = [
        ("/tmp/img1.jpg", "IMAGEM 1 (Frente)"),
        ("/tmp/img2.jpg", "IMAGEM 2 (Verso)")
    ]

    rotations = [0, 90, 180, 270]

    for img_path, img_name in images:
        print(f"\n{img_name}:")
        print("-" * 80)

        results = []
        for angle in rotations:
            result = rotate_and_test(img_path, angle, img_name)
            results.append(result)

        # Encontrar melhor rotação
        best = max(results, key=lambda x: x['char_count'])
        print(f"\n  ✓ MELHOR ROTAÇÃO: {best['angle']}° com {best['char_count']} caracteres")

        # Mostrar primeiros 500 chars do texto
        if best['text']:
            print(f"\n  ✓ Primeiros 500 caracteres extraídos:")
            print(f"    {best['text'][:500]}")

if __name__ == "__main__":
    main()
