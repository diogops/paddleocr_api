#!/usr/bin/env python3
"""
Testa 3 casos reais com diferentes qualidades de imagem
"""

import requests
import json
import time

# Casos de teste
CASOS = {
    "Caso 1 - Imagem Ruim (RG Rosenilda)": {
        "nome_esperado": "rosenilda luciene da cruz bessa",
        "mae_esperada": "gaudencia simoes dos santos",
        "cpf_esperado": "24065625572",
        "urls": [
            "https://incontadig.s3.amazonaws.com/2025/11/7/145e9169-dc3f-4a65-8f62-bb01dd902020/1k0v1up0i07dyqyiqv3oot_1762516091680.jpg",
            "https://incontadig.s3.amazonaws.com/2025/11/7/145e9169-dc3f-4a65-8f62-bb01dd902020/g55dncwk19doetcefiaiof_1762516108796.jpg"
        ]
    },
    "Caso 2 - Mesma Imagem (RG Maria Francisca)": {
        "nome_esperado": "maria francisca de jesus santos",
        "mae_esperada": "luzia francisca de jesus",
        "cpf_esperado": "05013678536",
        "urls": [
            "https://incontadig.s3.amazonaws.com/2025/11/7/16f90cc9-e57a-4e10-bdcf-670156b281f8/qfts3uxbktmauxbqzztags_1762518806175.jpg",
            "https://incontadig.s3.amazonaws.com/2025/11/7/16f90cc9-e57a-4e10-bdcf-670156b281f8/twczq1kc65q30t11d2lzh8_1762518811110.jpg"
        ]
    },
    "Caso 3 - Sem Qualidade (CNH PDF Tabiane)": {
        "nome_esperado": "tabiane luiza da silva barale",
        "mae_esperada": "maria de fatima luiza da silva",
        "cpf_esperado": "08450259614",
        "urls": [
            "https://incontadig.s3.amazonaws.com/2025/11/6/4ef2545c-3f94-44d2-81d3-467f9467176c/r9feaalowbsf16ug53ed8c_1762462423150.pdf"
        ]
    }
}

def normalizar_texto(texto):
    """Normaliza texto para comparação (lowercase, sem acentos)"""
    import unicodedata
    texto = texto.lower().strip()
    texto = unicodedata.normalize('NFKD', texto)
    texto = ''.join([c for c in texto if not unicodedata.combining(c)])
    return texto

def verificar_presenca(texto_extraido, texto_esperado, label):
    """Verifica se o texto esperado está presente no extraído"""
    texto_extraido_norm = normalizar_texto(texto_extraido)
    texto_esperado_norm = normalizar_texto(texto_esperado)

    # Verificar palavras individuais
    palavras_esperadas = texto_esperado_norm.split()
    palavras_encontradas = sum(1 for p in palavras_esperadas if p in texto_extraido_norm)

    presente = palavras_encontradas >= len(palavras_esperadas) * 0.7  # 70% das palavras

    if presente:
        return f"✓ {label}: ENCONTRADO ({palavras_encontradas}/{len(palavras_esperadas)} palavras)"
    else:
        return f"✗ {label}: NÃO ENCONTRADO ({palavras_encontradas}/{len(palavras_esperadas)} palavras)"

def testar_caso(nome_caso, caso):
    """Testa um caso específico"""
    print(f"\n{'='*80}")
    print(f"{nome_caso}")
    print('='*80)
    print(f"Esperado:")
    print(f"  Nome: {caso['nome_esperado']}")
    print(f"  Mãe: {caso['mae_esperada']}")
    print(f"  CPF: {caso['cpf_esperado']}")
    print(f"URLs: {len(caso['urls'])}")
    for i, url in enumerate(caso['urls'], 1):
        print(f"  {i}. {url.split('/')[-1]}")
    print()

    payload = {"urls": caso['urls']}

    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8000/ocr/extract",
            json=payload,
            timeout=120
        )
        elapsed = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            ocr_text = result.get("ocrText", "")
            extracted_fields = result.get("extractedFields", {})

            print(f"✓ Request bem-sucedido em {elapsed:.2f}s")
            print(f"\nTexto extraído: {len(ocr_text)} caracteres")
            print("-" * 80)
            print(ocr_text[:500] + ("..." if len(ocr_text) > 500 else ""))
            print("-" * 80)

            # Verificar campos extraídos
            print(f"\nCampos extraídos:")
            print(f"  documento_tipo: {extracted_fields.get('documento_tipo', 'N/A')}")
            print(f"  nome: {extracted_fields.get('nome', 'N/A')}")
            print(f"  mae: {extracted_fields.get('mae', 'N/A')}")
            print(f"  cpf: {extracted_fields.get('cpf', 'N/A')}")

            # Validação
            print(f"\nValidação:")
            print(verificar_presenca(ocr_text, caso['nome_esperado'], "Nome"))
            print(verificar_presenca(ocr_text, caso['mae_esperada'], "Mãe"))
            print(verificar_presenca(ocr_text, caso['cpf_esperado'], "CPF"))

            return {
                'sucesso': True,
                'tempo': elapsed,
                'chars': len(ocr_text),
                'texto': ocr_text,
                'campos': extracted_fields
            }
        else:
            print(f"✗ Erro HTTP {response.status_code}")
            print(f"Resposta: {response.text[:500]}")
            return {'sucesso': False, 'erro': f"HTTP {response.status_code}"}
    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT após {elapsed:.2f}s")
        return {'sucesso': False, 'erro': 'TIMEOUT'}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Erro após {elapsed:.2f}s: {e}")
        return {'sucesso': False, 'erro': str(e)}

if __name__ == "__main__":
    print("="*80)
    print("TESTE DE CASOS REAIS - DIFERENTES QUALIDADES DE IMAGEM")
    print("="*80)

    resultados = {}
    for nome_caso, caso in CASOS.items():
        resultado = testar_caso(nome_caso, caso)
        resultados[nome_caso] = resultado
        time.sleep(3)  # Pausa entre testes

    # Resumo final
    print(f"\n{'='*80}")
    print("RESUMO FINAL")
    print('='*80)
    for nome, resultado in resultados.items():
        if resultado['sucesso']:
            print(f"✓ {nome}: {resultado['chars']} chars em {resultado['tempo']:.2f}s")
        else:
            print(f"✗ {nome}: FALHOU - {resultado['erro']}")
    print('='*80)

    # Salvar resultados
    with open('test_casos_reais_result.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print("\nResultados salvos em: test_casos_reais_result.json")
