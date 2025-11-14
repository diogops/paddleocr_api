#!/usr/bin/env python3
"""
Testa cada documento individualmente
"""

import requests
import json
import time

# URLs individuais
URL1 = "https://incontadig.s3.amazonaws.com/2025/10/28/aedd4ce9-49ba-4bf2-a915-1394e9a238a7/nwend21esgrjdy1fsweenp_1761652969272.jpeg"
URL2 = "https://incontadig.s3.amazonaws.com/2025/10/28/aedd4ce9-49ba-4bf2-a915-1394e9a238a7/usgwl5wixck5ab0zcsjdd_1761652972718.jpeg"

def test_single_url(url, name):
    """Testa uma URL individual"""
    print(f"\n{'='*80}")
    print(f"TESTE: {name}")
    print('='*80)
    print(f"URL: {url.split('/')[-1]}")

    payload = {"urls": [url]}

    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8000/ocr/extract",
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            ocr_text = result.get("ocrText", "")
            print(f"✓ Sucesso em {elapsed:.2f}s")
            print(f"Texto extraído: {len(ocr_text)} caracteres")
            print(f"Primeiros 200 chars: {ocr_text[:200]}")
            return True
        else:
            print(f"✗ Erro HTTP {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT após {elapsed:.2f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Erro após {elapsed:.2f}s: {e}")
        return False

if __name__ == "__main__":
    print("TESTE INDIVIDUAL DE DOCUMENTOS")

    # Testar documento 1
    result1 = test_single_url(URL1, "Documento 1")

    # Aguardar um pouco
    time.sleep(2)

    # Testar documento 2
    result2 = test_single_url(URL2, "Documento 2")

    print(f"\n{'='*80}")
    print("RESUMO")
    print('='*80)
    print(f"Documento 1: {'✓ OK' if result1 else '✗ FALHOU'}")
    print(f"Documento 2: {'✓ OK' if result2 else '✗ FALHOU'}")
    print('='*80)
