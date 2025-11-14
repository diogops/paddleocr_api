#!/usr/bin/env python3
"""
Script para testar performance após reverter pré-processamento
Testa com 2 imagens JPEG (RG frente e verso)
"""

import requests
import json
import time

# URLs do RG do exemplo fornecido
TEST_URLS = [
    "https://incontadig.s3.amazonaws.com/2025/10/29/200bce4e-5e77-4922-ac20-426985d62ed7/70kcvzlxg9p34y1yi35f7q_1761740481842.jpeg",
    "https://incontadig.s3.amazonaws.com/2025/10/29/200bce4e-5e77-4922-ac20-426985d62ed7/gczojiaxmojep39o7n2vh_1761740498263.jpeg"
]

def test_performance():
    """Testa performance do OCR sem pré-processamento"""

    print("=" * 80)
    print("TESTE DE PERFORMANCE - SEM PRÉ-PROCESSAMENTO")
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
        print("RESULTADO")
        print("=" * 80)

        # Mostrar texto extraído
        ocr_text = result.get("ocrText", "")
        print(f"\nTexto extraído ({len(ocr_text)} caracteres):")
        print("-" * 80)
        print(ocr_text[:300] + "..." if len(ocr_text) > 300 else ocr_text)
        print("-" * 80)

        # Mostrar campos extraídos
        extracted_fields = result.get("extractedFields", {})
        print("\nCampos extraídos:")
        for key, value in extracted_fields.items():
            print(f"  {key}: {value}")

        # Mostrar estatísticas
        stats = result.get("stats", {})
        print("\nEstatísticas:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 80)
        print(f"PERFORMANCE")
        print("=" * 80)
        print(f"Tempo total: {elapsed_time:.2f}s")
        print(f"Tempo por documento: {elapsed_time/len(TEST_URLS):.2f}s")
        print("=" * 80)

        # Salvar resultado
        with open("/opt/paddleocr_api/test_performance_result.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\nResultado completo salvo em: test_performance_result.json")

        return elapsed_time

    else:
        print(f"\n✗ Erro na request: HTTP {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    test_performance()
