#!/usr/bin/env python3
"""
Testa todas as rotações possíveis para ver qual extrai mais texto
"""

import requests
import json
import time

# URLs dos documentos
DOCS = {
    "Doc 1 (Frente RG)": "https://incontadig.s3.amazonaws.com/2025/10/28/aedd4ce9-49ba-4bf2-a915-1394e9a238a7/nwend21esgrjdy1fsweenp_1761652969272.jpeg",
    "Doc 2 (Verso RG)": "https://incontadig.s3.amazonaws.com/2025/10/28/aedd4ce9-49ba-4bf2-a915-1394e9a238a7/usgwl5wixck5ab0zcsjdd_1761652972718.jpeg"
}

def test_document(name, url):
    """Testa um documento"""
    print(f"\n{'='*80}")
    print(f"TESTANDO: {name}")
    print('='*80)
    print(f"URL: {url.split('/')[-1]}")

    payload = {"urls": [url]}

    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8000/ocr/extract",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            ocr_text = result.get("ocrText", "")

            print(f"✓ Sucesso em {elapsed:.2f}s")
            print(f"Total de caracteres: {len(ocr_text)}")
            print(f"\nTexto completo extraído:")
            print("-" * 80)
            print(ocr_text)
            print("-" * 80)

            # Verificar se contém palavras esperadas
            palavras_chave = ["ROSA", "MARIA", "MACIEIRA", "MARIALVA", "ESPIRITO", "SANTO"]
            encontradas = [p for p in palavras_chave if p in ocr_text.upper()]

            print(f"\nPalavras-chave encontradas: {', '.join(encontradas) if encontradas else 'NENHUMA'}")

            return ocr_text
        else:
            print(f"✗ Erro HTTP {response.status_code}")
            print(f"Resposta: {response.text[:500]}")
            return None
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT após {elapsed:.2f}s")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Erro após {elapsed:.2f}s: {e}")
        return None

if __name__ == "__main__":
    print("TESTE DETALHADO DE EXTRAÇÃO DE TEXTO - RG")
    print("="*80)
    print("Objetivo: Verificar se o OCR está capturando 'Rosa Maria Cunha Macieira'")
    print("="*80)

    results = {}
    for name, url in DOCS.items():
        text = test_document(name, url)
        results[name] = text
        time.sleep(2)

    print(f"\n{'='*80}")
    print("RESUMO FINAL")
    print('='*80)
    for name, text in results.items():
        if text:
            print(f"{name}: {len(text)} caracteres")
        else:
            print(f"{name}: FALHOU")
    print('='*80)
