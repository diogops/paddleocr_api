#!/usr/bin/env python3
"""
Script para testar as melhorias de qualidade do OCR
Testa o documento de exemplo da CNH
"""

import requests
import json
import time

# URL do documento de exemplo fornecido pelo usuário
TEST_URL = "https://incontadig.s3.amazonaws.com/2025/11/6/4ef2545c-3f94-44d2-81d3-467f9467176c/r9feaalowbsf16ug53ed8c_1762462423150.pdf"

# Dados esperados do documento
EXPECTED_DATA = {
    "cpf": "08450259614",
    "ddn": "1988-06-08",
    "mae": "maria de fatima luiza da silva",
    "nome": "tabiane luiza da silva barale",
    "tipo": "cnh"
}

def test_ocr_extract():
    """Testa o endpoint /ocr/extract com o documento de exemplo"""

    print("=" * 80)
    print("TESTE DE MELHORIAS DE QUALIDADE DO OCR")
    print("=" * 80)
    print(f"\nDocumento de teste: {TEST_URL}")
    print(f"\nDados esperados:")
    for key, value in EXPECTED_DATA.items():
        print(f"  {key}: {value}")
    print("\n" + "=" * 80)

    # Preparar request
    payload = {
        "urls": [TEST_URL]
    }

    print("\nIniciando teste...")
    start_time = time.time()

    # Fazer request
    response = requests.post(
        "http://localhost:8000/ocr/extract",
        json=payload,
        timeout=300
    )

    elapsed_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()

        print(f"\n✓ Request bem-sucedido em {elapsed_time:.2f}s")
        print("\n" + "=" * 80)
        print("RESULTADO DO OCR")
        print("=" * 80)

        # Mostrar texto extraído
        ocr_text = result.get("ocrText", "")
        print(f"\nTexto extraído ({len(ocr_text)} caracteres):")
        print("-" * 80)
        print(ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text)
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

        # Validação
        print("\n" + "=" * 80)
        print("VALIDAÇÃO")
        print("=" * 80)

        validation_results = []

        # Verificar CPF
        extracted_cpf = extracted_fields.get("cpf", "").replace(".", "").replace("-", "")
        expected_cpf = EXPECTED_DATA["cpf"]
        cpf_match = extracted_cpf == expected_cpf
        validation_results.append(("CPF", cpf_match, expected_cpf, extracted_cpf))

        # Verificar nome (case insensitive)
        extracted_nome = extracted_fields.get("nome", "").lower()
        expected_nome = EXPECTED_DATA["nome"].lower()
        nome_match = expected_nome in extracted_nome or extracted_nome in expected_nome
        validation_results.append(("Nome", nome_match, EXPECTED_DATA["nome"], extracted_fields.get("nome", "")))

        # Verificar mãe (case insensitive)
        extracted_mae = extracted_fields.get("mae", "").lower()
        expected_mae = EXPECTED_DATA["mae"].lower()
        mae_match = expected_mae in extracted_mae or extracted_mae in expected_mae
        validation_results.append(("Mãe", mae_match, EXPECTED_DATA["mae"], extracted_fields.get("mae", "")))

        # Verificar tipo de documento
        doc_tipo = extracted_fields.get("documento_tipo", "").lower()
        tipo_match = "cnh" in doc_tipo or "habilitação" in doc_tipo or "habilitacao" in doc_tipo
        validation_results.append(("Tipo", tipo_match, EXPECTED_DATA["tipo"], extracted_fields.get("documento_tipo", "")))

        # Imprimir resultados de validação
        total_checks = len(validation_results)
        passed_checks = sum(1 for _, match, _, _ in validation_results if match)

        for field, match, expected, extracted in validation_results:
            status = "✓" if match else "✗"
            print(f"{status} {field}:")
            print(f"  Esperado: {expected}")
            print(f"  Extraído: {extracted}")

        print("\n" + "=" * 80)
        print(f"RESUMO: {passed_checks}/{total_checks} verificações passaram")
        print(f"Taxa de sucesso: {(passed_checks/total_checks)*100:.1f}%")
        print(f"Tempo de processamento: {elapsed_time:.2f}s")
        print("=" * 80)

        # Salvar resultado completo em arquivo
        with open("/opt/paddleocr_api/test_result.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\nResultado completo salvo em: test_result.json")

        return passed_checks, total_checks

    else:
        print(f"\n✗ Erro na request: HTTP {response.status_code}")
        print(response.text)
        return 0, 0

if __name__ == "__main__":
    test_ocr_extract()
