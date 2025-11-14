#!/usr/bin/env python3
"""
Script para testar as URLs específicas do problema reportado
"""

import requests
import json
import time

# URLs do problema reportado
TEST_URLS = [
    "https://incontadig.s3.amazonaws.com/2025/10/28/aedd4ce9-49ba-4bf2-a915-1394e9a238a7/nwend21esgrjdy1fsweenp_1761652969272.jpeg",
    "https://incontadig.s3.amazonaws.com/2025/10/28/aedd4ce9-49ba-4bf2-a915-1394e9a238a7/usgwl5wixck5ab0zcsjdd_1761652972718.jpeg"
]

def test_urls():
    """Testa as URLs específicas"""

    print("=" * 80)
    print("TESTE COM URLs ESPECÍFICAS DO PROBLEMA")
    print("=" * 80)
    print(f"\nNúmero de documentos: {len(TEST_URLS)}")
    print(f"\nURLs:")
    for i, url in enumerate(TEST_URLS, 1):
        print(f"  {i}. {url.split('/')[-1]}")
    print("\n" + "=" * 80)

    # Preparar request
    payload = {
        "urls": TEST_URLS
    }

    print("\nIniciando teste...")
    start_time = time.time()

    # Fazer request
    response = requests.post(
        "http://localhost:8000/ocr/extract",
        json=payload,
        timeout=120
    )

    elapsed_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()

        print(f"\n✓ Request bem-sucedido em {elapsed_time:.2f}s")
        print("\n" + "=" * 80)
        print("RESULTADO COMPLETO")
        print("=" * 80)

        # Mostrar texto extraído
        ocr_text = result.get("ocrText", "")
        print(f"\nocrText ({len(ocr_text)} caracteres):")
        print("-" * 80)
        print(ocr_text)
        print("-" * 80)

        # Mostrar campos extraídos
        extracted_fields = result.get("extractedFields", {})
        print("\nextractedFields:")
        print(json.dumps(extracted_fields, indent=2, ensure_ascii=False))

        # Mostrar estatísticas
        stats = result.get("stats", {})
        print("\nstats:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))

        print("\n" + "=" * 80)
        print("ANÁLISE")
        print("=" * 80)
        print(f"Tempo total: {elapsed_time:.2f}s")
        print(f"Tempo por documento: {elapsed_time/len(TEST_URLS):.2f}s")
        print(f"Total de caracteres: {len(ocr_text)}")
        print("\n" + "=" * 80)

        # Salvar resultado
        with open("/opt/paddleocr_api/test_specific_urls_result.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\nResultado completo salvo em: test_specific_urls_result.json")

        return result

    else:
        print(f"\n✗ Erro na request: HTTP {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    test_urls()
