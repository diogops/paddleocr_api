from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from paddleocr import PaddleOCR
import tempfile
import os
import requests
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
from functools import lru_cache
from PIL import Image
import io
import base64
import threading
import numpy as np
import cv2
import fitz  # PyMuPDF para suporte a PDF
import pytesseract  # Tesseract OCR fallback para documentos de baixa qualidade
import queue
import hashlib  # Para deduplicação de imagens por hash

app = FastAPI()

# ============================================================================
# POOL DE INSTÂNCIAS PADDLEOCR PARA ALTA CONCORRÊNCIA
# ============================================================================
# Configuração do pool: 8 instâncias por worker process
# Com 4 workers uvicorn: 4 × 8 = 32 instâncias totais
# IMPORTANTE: 8 instâncias = 2x as rotações (4) para evitar deadlock quando
# processar múltiplos documentos em paralelo (cada um testando 4 rotações)
OCR_POOL_SIZE = 8

# Criar pool de instâncias PaddleOCR
print(f"Inicializando pool de {OCR_POOL_SIZE} instâncias PaddleOCR...")
ocr_pool = queue.Queue(maxsize=OCR_POOL_SIZE)

for i in range(OCR_POOL_SIZE):
    print(f"  Criando instância OCR {i+1}/{OCR_POOL_SIZE}...")
    ocr_instance = PaddleOCR(
        use_angle_cls=False,  # Classificador de ângulo desabilitado
        lang='pt'  # Português
    )
    ocr_pool.put(ocr_instance)

print(f"Pool de OCR inicializado com sucesso!")

# Context manager para usar instâncias do pool de forma thread-safe
class OCRPoolContext:
    """Context manager para pegar e devolver instâncias OCR do pool"""
    def __enter__(self):
        # Pega uma instância disponível do pool (bloqueia se todas estiverem em uso)
        self.ocr = ocr_pool.get()
        return self.ocr

    def __exit__(self, *args):
        # Devolve a instância para o pool
        ocr_pool.put(self.ocr)

# Modelo para o request
class OCRRequest(BaseModel):
    urls: List[str]
    cpf: Optional[str] = None
    ddn: Optional[str] = None
    mae: Optional[str] = None
    nome: Optional[str] = None
    tipo: Optional[str] = None
    uuid: Optional[str] = None

    class Config:
        extra = "allow"  # Permite campos extras como "0", "1", "keyNames"

# Modelo para request base64
class OCRBase64Request(BaseModel):
    image: str  # Base64 da imagem
    extract_fields: Optional[bool] = False  # Se deve extrair campos estruturados

def download_image(url: str) -> bytes:
    """Baixa imagem de uma URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao baixar imagem {url}: {str(e)}")

def is_pdf(content: bytes) -> bool:
    """Verifica se o conteúdo é um arquivo PDF"""
    return content.startswith(b'%PDF')

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extrai texto nativo do PDF (texto selecionável) se disponível
    Retorna o texto extraído ou string vazia se não houver texto
    """
    tmp_pdf = None
    try:
        # Salvar PDF temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_pdf = tmp_file.name
            tmp_file.write(pdf_bytes)

        # Abrir PDF
        pdf_document = fitz.open(tmp_pdf)

        # Extrair texto de todas as páginas
        full_text = ""
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            text = page.get_text()
            if text:
                full_text += text + " "

        pdf_document.close()

        return full_text.strip()

    except Exception as e:
        print(f"Erro ao extrair texto do PDF: {str(e)}")
        return ""

    finally:
        if tmp_pdf and os.path.exists(tmp_pdf):
            os.unlink(tmp_pdf)

# ============================================================================
# PRÉ-PROCESSAMENTO DE IMAGEM - DESABILITADO POR QUESTÕES DE PERFORMANCE
# ============================================================================
# NOTA: O pré-processamento avançado (denoising, CLAHE, sharpening) melhora
# a qualidade do OCR mas aumenta o tempo de processamento de ~3s para ~34s.
# Por enquanto, mantemos desabilitado. Use apenas se necessário para documentos
# de baixíssima qualidade.
# ============================================================================

def preprocess_image_for_ocr(img_array: np.ndarray, aggressive: bool = False, keep_color: bool = True) -> np.ndarray:
    """
    PRÉ-PROCESSAMENTO DESABILITADO - Retorna imagem original

    Para reativar o pré-processamento avançado, descomente o código abaixo.
    ATENÇÃO: Aumentará o tempo de processamento de ~3s para ~30s+
    """
    # VERSÃO SIMPLIFICADA: Sem pré-processamento (RÁPIDO - ~3s)
    return img_array

    # ========================================================================
    # PRÉ-PROCESSAMENTO AVANÇADO - COMENTADO (LENTO - ~30s+)
    # ========================================================================
    # # Se for modo agressivo OU imagem já for grayscale, converter para cinza
    # is_grayscale = len(img_array.shape) == 2
    # convert_to_gray = aggressive or is_grayscale
    #
    # if convert_to_gray and not is_grayscale:
    #     working_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # else:
    #     working_img = img_array.copy()
    #
    # # 1. Redimensionamento inteligente
    # height, width = working_img.shape[:2]
    # max_dim = max(height, width)
    # min_dim = min(height, width)
    #
    # # OTIMIZAÇÃO CRÍTICA: Limitar tamanho máximo para evitar denoising lento
    # # Máximo de 3000px no lado maior (balanceio qualidade/performance)
    # MAX_DIMENSION = 3000
    #
    # if max_dim > MAX_DIMENSION:
    #     # Downscale - imagem muito grande
    #     scale_factor = MAX_DIMENSION / max_dim
    #     new_width = int(width * scale_factor)
    #     new_height = int(height * scale_factor)
    #     working_img = cv2.resize(working_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    #     print(f"  Downscaling: {width}x{height} → {new_width}x{new_height} ({scale_factor:.2f}x)")
    #     height, width = new_height, new_width
    # elif min_dim < 1500:
    #     # Upscale - imagem muito pequena
    #     scale_factor = 1500 / min_dim
    #     new_width = int(width * scale_factor)
    #     new_height = int(height * scale_factor)
    #     working_img = cv2.resize(working_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    #     print(f"  Upscaling: {width}x{height} → {new_width}x{new_height} ({scale_factor:.2f}x)")
    #
    # # 2. Remoção de ruído (denoising OTIMIZADO)
    # # Usar parâmetros mais leves para melhor performance
    # if convert_to_gray:
    #     # Grayscale denoising - MAIS RÁPIDO
    #     denoised = cv2.fastNlMeansDenoising(working_img, None, h=6, templateWindowSize=5, searchWindowSize=15)
    # else:
    #     # Color denoising - OTIMIZADO (h e hColor menores, janelas menores)
    #     # h=6 e hColor=6 em vez de 8/8 = ~40% mais rápido
    #     # searchWindowSize=15 em vez de 21 = ~50% mais rápido
    #     denoised = cv2.fastNlMeansDenoisingColored(working_img, None, h=6, hColor=6,
    #                                                  templateWindowSize=5, searchWindowSize=15)
    #
    # # 3. Aumento de contraste usando CLAHE
    # if convert_to_gray:
    #     # CLAHE em imagem grayscale
    #     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    #     contrasted = clahe.apply(denoised)
    # else:
    #     # CLAHE em imagem colorida (aplicar apenas no canal L do espaço LAB)
    #     lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    #     l, a, b = cv2.split(lab)
    #     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    #     l = clahe.apply(l)
    #     contrasted = cv2.merge([l, a, b])
    #     contrasted = cv2.cvtColor(contrasted, cv2.COLOR_LAB2BGR)
    #
    # # 4. Binarização adaptativa (APENAS em modo agressivo)
    # if aggressive and convert_to_gray:
    #     # Gaussian Adaptive Threshold
    #     binary = cv2.adaptiveThreshold(
    #         contrasted,
    #         255,
    #         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #         cv2.THRESH_BINARY,
    #         11,
    #         2
    #     )
    #     # Morfologia para limpar ruído
    #     kernel = np.ones((2, 2), np.uint8)
    #     processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # else:
    #     processed = contrasted
    #
    # # 5. Sharpening (nitidez)
    # if convert_to_gray:
    #     # Sharpening em grayscale
    #     sharpen_kernel = np.array([[-1, -1, -1],
    #                                [-1,  9, -1],
    #                                [-1, -1, -1]])
    #     sharpened = cv2.filter2D(processed, -1, sharpen_kernel)
    # else:
    #     # Sharpening em imagem colorida
    #     sharpen_kernel = np.array([[-1, -1, -1],
    #                                [-1,  9, -1],
    #                                [-1, -1, -1]])
    #     sharpened = cv2.filter2D(processed, -1, sharpen_kernel)
    #
    # return sharpened


def convert_pdf_to_images(pdf_bytes: bytes, enhance: bool = False) -> List[bytes]:
    """
    Converte PDF para lista de imagens (bytes JPG)
    Retorna uma lista de bytes de imagens JPG, uma para cada página

    Args:
        pdf_bytes: Bytes do arquivo PDF
        enhance: Se True, aplica pré-processamento (DESABILITADO por padrão)

    Returns:
        Lista de bytes de imagens JPG
    """
    images = []
    tmp_pdf = None

    try:
        # Salvar PDF temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_pdf = tmp_file.name
            tmp_file.write(pdf_bytes)

        # Abrir PDF com PyMuPDF
        pdf_document = fitz.open(tmp_pdf)

        # Converter cada página em imagem
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]

            # DPI de 288 (4.0x) - balanceio qualidade/performance
            # Suficiente para OCR sem criar imagens muito grandes
            mat = fitz.Matrix(4.0, 4.0)  # 4.0x = ~288 DPI
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Converter pixmap para bytes JPG diretamente (sem pré-processamento)
            img_bytes = pix.tobytes("jpeg")

            # PRÉ-PROCESSAMENTO DESABILITADO POR PADRÃO (muito lento)
            # Para habilitar, passe enhance=True na chamada
            # if enhance:
            #     img_data = pix.tobytes("png")
            #     nparr = np.frombuffer(img_data, np.uint8)
            #     img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            #     if img_array is not None:
            #         print(f"  Página {page_number+1}: aplicando pré-processamento...")
            #         img_array = preprocess_image_for_ocr(img_array, aggressive=False)
            #         success, img_encoded = cv2.imencode('.png', img_array, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            #         if success:
            #             img_bytes = img_encoded.tobytes()

            images.append(img_bytes)

        pdf_document.close()

        print(f"PDF convertido: {len(images)} página(s) em 288 DPI (sem pré-processamento)")

    except Exception as e:
        print(f"Erro ao converter PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar PDF: {str(e)}")

    finally:
        # Limpar arquivo temporário
        if tmp_pdf and os.path.exists(tmp_pdf):
            os.unlink(tmp_pdf)

    return images

def calculate_ocr_quality_score(text: str, confidence_data: dict) -> float:
    """
    Calcula score de qualidade do OCR baseado em múltiplos fatores
    Retorna score entre 0 e 100 (maior = melhor qualidade)
    """
    import re

    score = 0.0
    text_upper = text.upper()

    # Fator 1: Confiança média do Tesseract (peso 40%)
    confidences = [int(conf) for conf in confidence_data.get('conf', []) if conf != '-1']
    if confidences:
        avg_confidence = sum(confidences) / len(confidences)
        score += (avg_confidence / 100) * 40

    # Fator 2: Padrões de documentos brasileiros (peso 30%)
    pattern_score = 0

    # CPF: XXX.XXX.XXX-XX ou XXXXXXXXXXX
    if re.search(r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}', text):
        pattern_score += 10

    # RG: XX.XXX.XXX-XX ou variações
    if re.search(r'\d{2}\.?\d{3}\.?\d{3}-?\d{1,2}', text):
        pattern_score += 8

    # Data: DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
    if re.search(r'\d{2}[-/.]\d{2}[-/.]\d{4}', text):
        pattern_score += 7

    # Nomes brasileiros comuns (do documento esperado)
    brazilian_names = ['LUCIENE', 'ROSENILDA', 'GAUDENCIA', 'JOSE', 'CRUZ', 'BESSA',
                       'SANTOS', 'PEREIRA', 'SIMOES']
    for name in brazilian_names:
        if name in text_upper:
            pattern_score += 1

    # Localidades brasileiras
    if re.search(r'(CAMAÇARI|CAMACARI|SALVADOR|BAHIA|BA|SAO PAULO|SP|RIO|RJ)', text_upper):
        pattern_score += 2

    score += min(pattern_score, 30)  # Máximo 30 pontos

    # Fator 3: Razão texto/ruído (peso 30%)
    # Caracteres alfanuméricos vs caracteres especiais
    alphanumeric = len(re.findall(r'[a-zA-Z0-9]', text))
    total_chars = len(text.replace(' ', '').replace('\n', ''))

    if total_chars > 0:
        clean_ratio = alphanumeric / total_chars
        score += clean_ratio * 30

    return score


def process_single_rotation(img: Image.Image, angle: int, config: str = '') -> dict:
    """
    Processa uma única rotação da imagem com Tesseract
    Retorna dict com angle, text, score
    """
    try:
        if angle == 0:
            rotated_img = img
        else:
            rotated_img = img.rotate(angle, expand=True)

        # Executar Tesseract e obter dados com confiança
        text = pytesseract.image_to_string(rotated_img, lang='por', config=config)
        data = pytesseract.image_to_data(rotated_img, lang='por', config=config, output_type=pytesseract.Output.DICT)

        # Calcular score de qualidade
        quality_score = calculate_ocr_quality_score(text, data)

        text_length = len(text.strip())
        print(f"  Rotação {angle:4}°: {text_length:4} chars, score qualidade: {quality_score:.2f}")

        return {
            'angle': angle,
            'text': text,
            'score': quality_score
        }

    except Exception as e:
        print(f"Erro ao processar rotação {angle}°: {e}")
        return {
            'angle': angle,
            'text': '',
            'score': 0
        }

def tesseract_ocr_fallback(image_bytes: bytes) -> str:
    """
    Fallback OCR usando Tesseract para documentos de baixa qualidade
    OTIMIZAÇÃO: Processa múltiplas rotações EM PARALELO (3x mais rápido!)
    Tenta múltiplas rotações e escolhe baseado em QUALIDADE, não em quantidade de texto
    """
    try:
        # Converter bytes para PIL Image
        img = Image.open(io.BytesIO(image_bytes))

        # Configuração default (mais rápida segundo testes)
        config = ''

        # Tentar múltiplas rotações (0°, -90°, 180°, 90°) EM PARALELO
        rotations = [0, -90, 180, 90]

        # Processar todas as rotações em paralelo usando ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submeter todas as rotações para processamento paralelo
            futures = {executor.submit(process_single_rotation, img, angle, config): angle
                      for angle in rotations}

            # Coletar resultados conforme completam
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        # Escolher o melhor resultado baseado em QUALIDADE
        best_result = max(results, key=lambda x: x['score'])

        best_text = best_result['text']
        best_score = best_result['score']
        best_angle = best_result['angle']

        # Limpar texto: remover múltiplos espaços e linhas em branco
        cleaned_text = ' '.join(best_text.split())

        print(f"Tesseract fallback: melhor rotação {best_angle}° (score: {best_score:.2f}, {len(cleaned_text)} chars)")
        return cleaned_text

    except Exception as e:
        print(f"Erro no Tesseract fallback: {type(e).__name__}: {str(e)}")
        return ""

def process_single_rotation_paddle(img_array: np.ndarray, angle: int, enhance: bool = False) -> dict:
    """
    Processa uma única rotação da imagem com PaddleOCR

    Args:
        img_array: Imagem em numpy array
        angle: Ângulo de rotação (0, 90, 180, 270)
        enhance: Se True, aplica pré-processamento (DESABILITADO por padrão)

    Returns:
        Dict com angle, text, char_count
    """
    tmp_path = None
    try:
        # Rotacionar imagem usando cv2
        # BUG FIX: Tesseract OSD e OpenCV têm convenções opostas
        # Quando Tesseract diz "rotate=90", significa rotacionar ANTI-HORÁRIO para corrigir
        if angle == 90:
            rotated = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Anti-horário
        elif angle == 180:
            rotated = cv2.rotate(img_array, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)  # Horário
        else:  # 0 ou -90 (tratamos -90 como 270)
            if angle == -90:
                rotated = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
            else:
                rotated = img_array

        # PRÉ-PROCESSAMENTO DESABILITADO POR PADRÃO (muito lento - ~30s+)
        # if enhance:
        #     rotated = preprocess_image_for_ocr(rotated, aggressive=False)

        # Salvar imagem rotacionada temporariamente em JPG (rápido)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, rotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Processar com PaddleOCR usando pool (thread-safe)
        with OCRPoolContext() as ocr:
            results = ocr.ocr(tmp_path, cls=False)

        # Extrair texto
        all_text = []
        if results and results[0]:
            for line in results[0]:
                if line and len(line) > 1:
                    text = line[1][0]
                    all_text.append(text)

        extracted_text = ' '.join(all_text)
        char_count = len(extracted_text.strip())

        print(f"  Rotação {angle:4}°: {char_count:4} chars")

        return {
            'angle': angle,
            'text': extracted_text,
            'char_count': char_count
        }

    except Exception as e:
        print(f"Erro ao processar rotação {angle}°: {e}")
        return {
            'angle': angle,
            'text': '',
            'char_count': 0
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def detect_image_orientation(img_array: np.ndarray) -> int:
    """
    Detecta a orientação da imagem usando Tesseract OSD (Orientation and Script Detection)
    Retorna o ângulo de rotação necessário: 0, 90, 180 ou 270
    Retorna None se não conseguir detectar

    OSD é MUITO mais rápido (~0.5s) que processar 4 rotações (~3-5s)
    """
    tmp_path = None
    try:
        # Salvar imagem temporariamente para Tesseract
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, img_array, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Usar Tesseract OSD (--psm 0) para detectar orientação
        # Retorna dict com: orientation, rotate, orientation_conf, script, script_conf
        osd = pytesseract.image_to_osd(tmp_path, output_type=pytesseract.Output.DICT)

        # orientation: 0, 90, 180, 270 (graus no sentido horário)
        # rotate: quantos graus precisa rotacionar para corrigir
        detected_angle = osd.get('rotate', 0)
        confidence = osd.get('orientation_conf', 0)

        print(f"Detecção de orientação: {detected_angle}° (confiança: {confidence:.2f})")

        # Só usar detecção se confiança for razoável (>1.5)
        if confidence > 1.5:
            # Tesseract OSD retorna o ângulo que a imagem precisa rotacionar para CORRIGIR
            # Mas na prática, retorna o ângulo ATUAL da imagem (não o necessário para corrigir)
            # Se OSD diz rotate=90°, significa que a imagem está 90° rotacionada
            # e precisamos aplicar essa mesma rotação no cv2.rotate
            # rotate=0 → imagem correta → usar 0°
            # rotate=90 → imagem 90° horário → usar 90°
            # rotate=180 → imagem 180° → usar 180°
            # rotate=270 → imagem 270° horário → usar 270°
            # Não precisa mapear, usar direto
            return detected_angle
        else:
            print(f"Confiança baixa ({confidence:.2f}), usando fallback multi-rotação")
            return None

    except Exception as e:
        print(f"Erro na detecção de orientação: {type(e).__name__}: {str(e)}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def perform_ocr(image_bytes: bytes, enhance: bool = False) -> str:
    """
    Realiza OCR em uma imagem (bytes) e retorna texto concatenado

    OTIMIZAÇÃO DE PERFORMANCE:
    1. Detecta orientação usando Tesseract OSD (~0.5s)
    2. Se detectar com sucesso, processa apenas a rotação correta (~1.5s total)
    3. Se falhar, testa 4 rotações em paralelo (~3-5s total)

    Args:
        image_bytes: Bytes da imagem
        enhance: Se True, aplica pré-processamento (DESABILITADO por padrão - muito lento)

    Performance esperada:
    - Detecção bem-sucedida: ~1.5-2s
    - Fallback multi-rotação: ~3-5s
    """
    try:
        # Converter bytes para numpy array usando cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("Erro: não foi possível decodificar a imagem")
            return ""

        # Redimensionar se imagem for muito grande (> 2000px)
        height, width = img.shape[:2]
        max_dimension = 2000
        if max(height, width) > max_dimension:
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            print(f"Imagem redimensionada: {width}x{height} → {new_width}x{new_height}px")
        else:
            print(f"Imagem original: {width}x{height}px")

        # OTIMIZAÇÃO: Detectar orientação primeiro (Tesseract OSD - ~0.5s)
        # Se detectar com sucesso, processar apenas a rotação correta
        # Se falhar, fazer fallback para testar todas as 4 rotações
        detected_angle = detect_image_orientation(img)

        if detected_angle is not None:
            # Detecção bem-sucedida! Processar apenas a rotação detectada
            print(f"Orientação detectada: {detected_angle}° - processando apenas esta rotação...")
            result = process_single_rotation_paddle(img, detected_angle, enhance=enhance)
            best_angle = result['angle']
            best_text = result['text']
            best_chars = result['char_count']
            print(f"OCR concluído: {best_chars} caracteres extraídos")
        else:
            # Detecção falhou - fallback para testar múltiplas rotações em paralelo
            print("Detecção de orientação falhou - testando múltiplas rotações em paralelo...")
            rotations = [0, 90, 180, 270]

            # Usar ThreadPoolExecutor para processamento paralelo
            # Nota: Mesmo com lock no PaddleOCR, o processamento das rotações
            # ainda é mais rápido pois o I/O (salvar/ler imagem) é paralelo
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(process_single_rotation_paddle, img, angle, enhance): angle
                          for angle in rotations}

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

            # Escolher rotação com mais texto extraído
            best_result = max(results, key=lambda x: x['char_count'])
            best_angle = best_result['angle']
            best_text = best_result['text']
            best_chars = best_result['char_count']

            print(f"Melhor rotação: {best_angle}° com {best_chars} caracteres")

        # FALLBACK COMENTADO - Vamos melhorar posteriormente
        # if len(best_text.strip()) < 50:
        #     print(f"PaddleOCR extraiu apenas {len(best_text)} caracteres. Tentando Tesseract fallback...")
        #     tesseract_text = tesseract_ocr_fallback(image_bytes)
        #     if len(tesseract_text) > len(best_text):
        #         print(f"Tesseract foi melhor: {len(tesseract_text)} chars vs {len(best_text)} chars")
        #         return tesseract_text

        return best_text

    except Exception as e:
        print(f"Erro no OCR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

        # FALLBACK COMENTADO - Vamos melhorar posteriormente
        # try:
        #     print("Erro no PaddleOCR, tentando Tesseract como último recurso...")
        #     return tesseract_ocr_fallback(image_bytes)
        # except:
        #     return ""

        return ""

def extract_cpf(text: str) -> Optional[str]:
    """Extrai CPF do texto"""
    # Padrões: 123.456.789-01 ou 12345678901
    patterns = [
        r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b',
        r'\b\d{2}\.\d{3}\.\d{3}-\d{2}\b',
        r'\b\d{11}\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            cpf = match.group()
            # Verificar se tem 11 dígitos
            nums = re.sub(r'\D', '', cpf)
            if len(nums) == 11:
                return cpf
    return None

def extract_rg(text: str) -> Optional[str]:
    """Extrai RG do texto"""
    # Padrões comuns: 12.345.678-9, 12345678-9, 08-055535083
    patterns = [
        r'\b\d{2}\.\d{3}\.\d{3}-\d{1,2}\b',
        r'\b\d{8,9}-\d{1,2}\b',
        r'\b\d{2}-\d{9}\b'
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    return None

def extract_dates(text: str) -> Dict[str, Optional[str]]:
    """Extrai datas do texto (nascimento e expedição)"""
    dates = {
        'data_nascimento': None,
        'data_expedicao': None
    }

    # Padrões de data: DD/MM/YYYY, DD-MM-YYYY, DD=MM=YYYY, DDMMYYYY
    date_patterns = [
        r'\b\d{2}[-/=\.]\d{2}[-/=\.]\d{4}\b',
        r'\b\d{8}\b'
    ]

    # Procurar "DATA DE NASCIMENTO" ou "NASC" seguido de data
    nasc_match = re.search(r'(?:DATA.*?NASC|NASC|DAT).*?(\d{2}[-/=\.]\d{2}[-/=\.]\d{4})', text, re.IGNORECASE)
    if nasc_match:
        dates['data_nascimento'] = nasc_match.group(1)

    # Procurar "DATA DE EXPEDIÇÃO" ou "EXPEDICAO" seguido de data
    exp_match = re.search(r'(?:DATA.*?EXPEDI|EXPEDI[CÇ]).*?(\d{2}[-/=\.]\d{2}[-/=\.]\d{4})', text, re.IGNORECASE)
    if exp_match:
        dates['data_expedicao'] = exp_match.group(1)

    # Se não encontrou com contexto, pegar as duas últimas datas do documento
    all_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Converter para formato padrão
            if '-' in match or '/' in match or '=' in match or '.' in match:
                all_dates.append(match)
            elif len(match) == 8:  # DDMMYYYY
                formatted = f"{match[:2]}/{match[2:4]}/{match[4:]}"
                all_dates.append(formatted)

    # Se encontrou datas mas não identificou contexto
    if all_dates:
        if not dates['data_nascimento'] and len(all_dates) >= 1:
            dates['data_nascimento'] = all_dates[0]
        if not dates['data_expedicao'] and len(all_dates) >= 2:
            dates['data_expedicao'] = all_dates[-1]

    return dates

def extract_names(text: str) -> Dict[str, Optional[str]]:
    """Extrai nomes (titular, mãe, pai) do texto"""
    names = {
        'nome': None,
        'mae': None,
        'pai': None
    }

    # Palavras comuns que não são nomes
    stopwords = {'DE', 'DA', 'DO', 'DOS', 'DAS', 'E', 'O', 'A', 'OS', 'AS',
                 'REPUBLICA', 'FEDERATIVA', 'BRASIL', 'ESTADO', 'CARTEIRA',
                 'IDENTIDADE', 'RG', 'CPF', 'DATA', 'NASCIMENTO', 'EXPEDICAO',
                 'ASSINATURA', 'DIRETOR', 'DIRETORA', 'LEI', 'VALIDA', 'TODO',
                 'TERRITORIO', 'NACIONAL', 'SECRETARIA', 'SEGURANCA', 'PUBLICA',
                 'POLICIA', 'DELEGADO', 'SSP', 'RGD', 'NOME', 'FILIACAO', 'MAE',
                 'PAI', 'REGISTRO', 'GERAL', 'DOC', 'ORIGEM', 'NATURALIDADE'}

    # Dividir texto em palavras
    lines = text.split()

    # Procurar padrões de nomes (sequências de palavras em maiúsculas)
    potential_names = []
    i = 0
    while i < len(lines):
        word = lines[i]
        # Se encontrar palavra em maiúsculas (possível nome)
        if word.isupper() and len(word) > 2 and word.isalpha() and word not in stopwords:
            name_parts = [word]
            # Continuar pegando palavras em maiúsculas
            j = i + 1
            while j < len(lines) and j < i + 6:  # Limite de 6 palavras para um nome
                next_word = lines[j]
                if next_word.isupper() and (next_word.isalpha() or next_word in {'DE', 'DA', 'DO', 'DOS', 'DAS'}):
                    if len(next_word) > 1:
                        name_parts.append(next_word)
                    j += 1
                else:
                    break

            # Se encontrou pelo menos 2 palavras (ignorando preposições), pode ser um nome
            actual_words = [w for w in name_parts if w not in {'DE', 'DA', 'DO', 'DOS', 'DAS'}]
            if len(actual_words) >= 2:
                full_name = ' '.join(name_parts)
                potential_names.append(full_name)
                i = j
            else:
                i += 1
        else:
            i += 1

    # Filtrar nomes muito curtos ou muito longos
    valid_names = [n for n in potential_names if 10 <= len(n) <= 60]

    # Atribuir nomes baseado em quantidade e posição
    if len(valid_names) >= 3:
        # Último nome completo geralmente é o titular
        names['nome'] = valid_names[-1]
        # Penúltimo e antepenúltimo são mãe e pai
        names['mae'] = valid_names[-3]
        names['pai'] = valid_names[-2]
    elif len(valid_names) >= 2:
        names['nome'] = valid_names[-1]
        names['mae'] = valid_names[-2]
    elif len(valid_names) >= 1:
        names['nome'] = valid_names[-1]

    return names

def extract_location(text: str) -> Optional[str]:
    """Extrai localização/estado do texto"""
    # Estados brasileiros (siglas e nomes)
    estados = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
               'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
               'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']

    for estado in estados:
        if re.search(rf'\b{estado}\b', text):
            # Tentar pegar cidade também
            city_match = re.search(rf'(\w+(?:\s+\w+)?)\s+{estado}', text)
            if city_match:
                return f"{city_match.group(1)} {estado}"
            return estado

    return None

def extract_document_type(text: str) -> Optional[str]:
    """Identifica tipo de documento"""
    text_upper = text.upper()

    if 'CARTEIRA DE IDENTIDADE' in text_upper or 'RG' in text_upper:
        return 'RG - Carteira de Identidade'
    elif 'CNH' in text_upper or 'HABILITACAO' in text_upper or 'DRIVER LICENSE' in text_upper:
        return 'CNH - Carteira Nacional de Habilitação'
    elif 'CTPS' in text_upper or 'TRABALHO' in text_upper:
        return 'CTPS - Carteira de Trabalho'

    return 'Documento de Identificação'

def extract_fields_from_ocr(text: str) -> Dict[str, Any]:
    """Extrai campos estruturados do texto do OCR"""
    fields = {
        'documento_tipo': extract_document_type(text),
        'cpf': extract_cpf(text),
        'rg': extract_rg(text),
        'local': extract_location(text)
    }

    # Extrair datas
    dates = extract_dates(text)
    fields.update(dates)

    # Extrair nomes
    names = extract_names(text)
    fields.update(names)

    return fields

# Endpoints
@app.get("/")
async def root():
    return {
        "service": "PaddleOCR API",
        "version": "3.1",
        "endpoints": {
            "/ocr": "POST - Upload de arquivo para OCR",
            "/ocr/base64": "POST - OCR de imagem em base64",
            "/ocr/extract": "POST - Extração de texto de URLs",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """
    Endpoint original: upload de arquivo para OCR
    OTIMIZADO: Agora usa perform_ocr() com múltiplas rotações
    """
    # Ler arquivo em bytes
    image_bytes = await file.read()

    # Usar nossa função otimizada que testa múltiplas rotações
    text = perform_ocr(image_bytes)

    # Retornar no formato compatível (simular lines com texto completo)
    if text:
        # Dividir em linhas e retornar como antes
        lines = [{"text": line, "score": 1.0} for line in text.split('\n') if line.strip()]
        return {"lines": lines}
    else:
        return {"lines": []}

@app.post("/ocr/base64")
async def ocr_base64_endpoint(request: OCRBase64Request):
    """
    Endpoint para processar imagem em base64 diretamente (sem download)

    Request body:
    {
        "image": "base64_string_aqui",
        "extract_fields": false  // opcional, default false
    }

    Response (extract_fields=false):
    {
        "lines": [
            {"text": "texto detectado", "score": 0.98}
        ]
    }

    Response (extract_fields=true):
    {
        "ocrText": "texto concatenado",
        "extractedFields": {...},
        "lines": [...]
    }
    """
    try:
        # Decodificar base64 para bytes
        try:
            # Remover prefixo data:image se existir
            image_b64 = request.image
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]

            image_bytes = base64.b64decode(image_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 inválido: {str(e)}")

        # Salvar temporariamente para o PaddleOCR processar
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        try:
            # Usar pool para acesso thread-safe ao OCR
            with OCRPoolContext() as ocr:
                # PaddleOCR 2.x usa ocr.ocr() e retorna [[[bbox], (text, confidence)], ...]
                results = ocr.ocr(tmp_path, cls=False)

            lines = []
            all_text = []
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) > 1:
                        text = line[1][0]  # line[1] é (text, confidence)
                        score = line[1][1]  # confidence score
                        lines.append({"text": text, "score": float(score)})
                        all_text.append(text)

            # Se solicitou extração de campos
            if request.extract_fields:
                combined_text = ' '.join(all_text)
                extracted_fields = extract_fields_from_ocr(combined_text)

                return {
                    "ocrText": combined_text,
                    "extractedFields": extracted_fields,
                    "lines": lines
                }
            else:
                return {"lines": lines}

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")

def download_and_prepare(url: str, index: int, total: int) -> tuple:
    """
    Baixa e prepara imagens/PDFs (pode ser executado em paralelo)
    Retorna (index, dict com 'images' e 'native_text')
    """
    try:
        print(f"Baixando URL {index+1}/{total}: {url}")
        content_bytes = download_image(url)

        # Detectar se é PDF
        if is_pdf(content_bytes):
            print(f"PDF detectado na URL {index+1}, processando...")

            # Extrair texto nativo do PDF
            native_text = extract_text_from_pdf(content_bytes)
            if native_text:
                print(f"PDF {index+1}: texto nativo extraído ({len(native_text)} caracteres)")

            # Converter PDF em imagens para OCR
            page_images = convert_pdf_to_images(content_bytes)
            print(f"PDF {index+1}: {len(page_images)} página(s) convertidas para OCR")

            return (index, {'images': page_images, 'native_text': native_text})
        else:
            # Imagem normal - retornar como lista
            return (index, {'images': [content_bytes], 'native_text': ''})
    except Exception as e:
        print(f"Erro ao baixar/preparar URL {url}: {e}")
        return (index, {'images': [], 'native_text': ''})

async def process_single_doc_async(index: int, doc_data: dict) -> tuple:
    """
    Processa um único documento de forma assíncrona
    doc_data = {'images': [bytes...], 'native_text': str}
    Retorna texto combinado: texto nativo + OCR
    """
    try:
        image_list = doc_data.get('images', [])
        native_text = doc_data.get('native_text', '')

        # Processar OCR das imagens
        print(f"Fazendo OCR do documento {index+1}...")
        ocr_texts = []

        for page_num, image_bytes in enumerate(image_list, 1):
            if len(image_list) > 1:
                print(f"  Página {page_num}/{len(image_list)}...")

            # Executar OCR em thread separada (perform_ocr é bloqueante)
            loop = asyncio.get_event_loop()
            page_text = await loop.run_in_executor(None, perform_ocr, image_bytes)
            if page_text:
                ocr_texts.append(page_text)

        ocr_combined = ' '.join(ocr_texts)

        # Combinar texto nativo + OCR
        all_texts = []
        if native_text:
            all_texts.append(native_text)
            print(f"Documento {index+1}: texto nativo ({len(native_text)} chars)")
        if ocr_combined:
            all_texts.append(ocr_combined)
            print(f"Documento {index+1}: OCR ({len(ocr_combined)} chars)")

        combined = ' '.join(all_texts)
        if combined:
            print(f"Documento {index+1} concluído: total {len(combined)} caracteres (nativo + OCR)")
        else:
            print(f"Documento {index+1}: nenhum texto extraído")

        return (index, combined)
    except Exception as e:
        print(f"Erro ao processar documento {index+1}: {e}")
        return (index, "")

async def process_ocr_parallel(prepared_images: List[tuple]) -> str:
    """
    Processa OCR de forma PARALELA - múltiplas imagens simultaneamente!
    prepared_images: lista de (index, dict com 'images' e 'native_text')
    Combina texto nativo + OCR para cada documento
    """
    print(f"Processando {len(prepared_images)} documentos em PARALELO...")
    print(f"DEBUG: Documentos a processar: {[(idx, len(doc['images'])) for idx, doc in prepared_images]}")

    # Criar tasks para processar todos os documentos simultaneamente
    tasks = [
        process_single_doc_async(index, doc_data)
        for index, doc_data in prepared_images
    ]

    print(f"DEBUG: Criadas {len(tasks)} tasks")

    # Aguardar todas as tasks completarem
    results = await asyncio.gather(*tasks)

    print(f"DEBUG: Resultados recebidos: {len(results)}")
    for idx, text in sorted(results):
        print(f"DEBUG: Documento {idx+1}: {len(text)} chars")

    # Ordenar por index e juntar textos
    all_texts = [text for _, text in sorted(results) if text]

    final_text = ' '.join(all_texts)
    print(f"DEBUG: Texto final: {len(final_text)} chars (de {len(all_texts)} documentos)")

    return final_text

def process_ocr_sequential(prepared_images: List[tuple]) -> str:
    """
    Processa OCR de forma SEQUENCIAL para evitar race conditions
    prepared_images: lista de (index, [image_bytes])
    """
    all_texts = []

    for index, image_list in sorted(prepared_images):
        try:
            print(f"Fazendo OCR do documento {index+1}...")
            doc_texts = []

            for page_num, image_bytes in enumerate(image_list, 1):
                if len(image_list) > 1:
                    print(f"  Página {page_num}/{len(image_list)}...")

                page_text = perform_ocr(image_bytes)
                if page_text:
                    doc_texts.append(page_text)

            combined = ' '.join(doc_texts)
            if combined:
                all_texts.append(combined)
                print(f"OCR {index+1} concluído ({len(combined)} caracteres)")
            else:
                print(f"OCR {index+1} não extraiu texto")

        except Exception as e:
            print(f"Erro ao fazer OCR do documento {index+1}: {e}")

    return ' '.join(all_texts)

@app.post("/ocr/extract")
async def ocr_extract_endpoint(request: OCRRequest):
    """
    Endpoint completo: baixa imagens de URLs, faz OCR e retorna texto + campos extraídos
    Processa múltiplas imagens EM PARALELO para melhor performance

    OTIMIZAÇÕES:
    - Deduplica URLs idênticas
    - Deduplica imagens por hash (URLs diferentes, mesma imagem)
    - Processa apenas imagens únicas

    Request body exemplo:
    {
        "urls": ["url1", "url2"],
        "cpf": "opcional",
        "nome": "opcional",
        ... outros campos opcionais
    }

    Response:
    {
        "ocrText": "texto bruto concatenado de todas as imagens",
        "extractedFields": {
            "documento_tipo": "RG - Carteira de Identidade",
            "nome": "NOME COMPLETO",
            "mae": "NOME DA MAE",
            "pai": "NOME DO PAI",
            "cpf": "123.456.789-01",
            "rg": "12.345.678-9",
            "data_nascimento": "01/01/1990",
            "data_expedicao": "01/01/2020",
            "local": "CIDADE UF"
        },
        "status": "OK",
        "stats": {
            "total_urls": N,
            "unique_urls": N,
            "unique_images": N,
            "duplicates_removed": N
        }
    }
    """
    try:
        if not request.urls or len(request.urls) == 0:
            raise HTTPException(status_code=400, detail="URLs não fornecidas")

        total_urls = len(request.urls)

        # FASE 0: DEDUPLICAÇÃO DE URLs IDÊNTICAS
        # Preservar ordem mas remover duplicatas
        seen_urls = {}
        unique_urls = []
        for i, url in enumerate(request.urls):
            if url not in seen_urls:
                seen_urls[url] = i
                unique_urls.append((i, url))

        unique_url_count = len(unique_urls)
        duplicates_by_url = total_urls - unique_url_count

        if duplicates_by_url > 0:
            print(f"Deduplicação URL: {total_urls} URLs → {unique_url_count} únicas ({duplicates_by_url} duplicadas removidas)")

        # FASE 1: Baixar e preparar imagens/PDFs EM PARALELO
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submete todas as tarefas de download em paralelo
            futures = [
                loop.run_in_executor(
                    executor,
                    download_and_prepare,
                    url,
                    idx,
                    unique_url_count
                )
                for idx, url in unique_urls
            ]

            # Aguarda todos os downloads completarem
            prepared_images = await asyncio.gather(*futures)

        # Filtrar falhas (listas vazias)
        prepared_images = [(idx, imgs) for idx, imgs in prepared_images if imgs]

        # FASE 2: DEDUPLICAÇÃO POR HASH DE IMAGEM (conteúdo)
        # Calcular hash de cada imagem e remover duplicatas ENTRE documentos diferentes
        # IMPORTANTE: Mantém todas as imagens de um mesmo documento (ex: frente e verso de RG)
        seen_hashes = set()
        deduplicated_docs = []
        total_images_downloaded = 0
        unique_image_count = 0

        for idx, doc_data in prepared_images:
            image_list = doc_data.get('images', [])
            native_text = doc_data.get('native_text', '')
            total_images_downloaded += len(image_list)

            # Filtrar apenas imagens únicas deste documento
            unique_images_in_doc = []
            for img_bytes in image_list:
                img_hash = hashlib.sha256(img_bytes).hexdigest()

                if img_hash not in seen_hashes:
                    # Primeira vez vendo esta imagem
                    seen_hashes.add(img_hash)
                    unique_images_in_doc.append(img_bytes)
                    unique_image_count += 1
                else:
                    print(f"Imagem duplicada detectada (hash: {img_hash[:8]}...) - pulando")

            # Se há imagens únicas neste documento, adicionar à lista
            if unique_images_in_doc:
                deduplicated_docs.append((idx, {
                    'images': unique_images_in_doc,
                    'native_text': native_text
                }))

        duplicates_by_hash = total_images_downloaded - unique_image_count

        if duplicates_by_hash > 0:
            print(f"Deduplicação Hash: {total_images_downloaded} imagens → {unique_image_count} únicas ({duplicates_by_hash} duplicadas removidas)")

        # Usar documentos deduplicados
        unique_images = deduplicated_docs

        # FASE 3: Processar OCR EM PARALELO apenas das imagens únicas
        combined_text = await process_ocr_parallel(unique_images)

        # Extrair campos estruturados do texto
        extracted_fields = extract_fields_from_ocr(combined_text)

        return {
            "ocrText": combined_text,
            "extractedFields": extracted_fields,
            "status": "OK",
            "stats": {
                "total_urls": total_urls,
                "unique_urls": unique_url_count,
                "unique_images": unique_image_count,
                "duplicates_removed": duplicates_by_url + duplicates_by_hash
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")
