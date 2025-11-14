#!/usr/bin/env python3
"""
Script de teste para o endpoint /ocr/base64
"""

import requests
import base64
import json

# URL do servidor (ajuste conforme necessário)
BASE_URL = "http://localhost:8000"

def test_base64_ocr(image_path: str, extract_fields: bool = False):
    """
    Testa o endpoint /ocr/base64 com uma imagem local

    Args:
        image_path: Caminho para a imagem
        extract_fields: Se deve extrair campos estruturados
    """
    # Ler imagem e converter para base64
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    # Preparar request
    payload = {
        "image": image_b64,
        "extract_fields": extract_fields
    }

    # Fazer requisição
    print(f"Enviando imagem {image_path} para OCR...")
    response = requests.post(f"{BASE_URL}/ocr/base64", json=payload)

    # Exibir resultado
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Sucesso!")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n❌ Erro: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python test_base64.py <caminho_da_imagem> [extract_fields]")
        print("\nExemplos:")
        print("  python test_base64.py minha_imagem.jpg")
        print("  python test_base64.py documento.png true")
        sys.exit(1)

    image_path = sys.argv[1]
    extract_fields = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'

    test_base64_ocr(image_path, extract_fields)
