from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Any, Dict
from paddleocr import PaddleOCR
import tempfile
import os
import requests
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import lru_cache
from PIL import Image
import io
import base64
import threading
import numpy as np
import cv2
import fitz  # PyMuPDF para suporte a PDF

# Tesseract removido - usando apenas PaddleOCR racing
import queue
import hashlib  # Para deduplicação de imagens por hash

app = FastAPI()

# ============================================================================
# POOL DE INSTÂNCIAS PADDLEOCR PARA ALTA CONCORRÊNCIA
# ============================================================================


# Detectar automaticamente se há GPU disponível
def check_gpu_available():
    """
    Verifica se há GPU disponível para PaddlePaddle
    Retorna True apenas se GPU estiver disponível E funcional

    IMPORTANTE: Se GPU falhar, retorna False para usar CPU seguramente
    """
    try:
        import paddle
        import os

        # Verificar variáveis de ambiente CUDA
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(
            f"CUDA_VISIBLE_DEVICES: {cuda_visible if cuda_visible else 'não configurado'}"
        )
        print(
            f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'não configurado')}"
        )

        gpu_count = paddle.device.cuda.device_count()
        has_gpu = gpu_count > 0

        if has_gpu:
            print(f"✅ GPU detectada! {gpu_count} GPU(s) disponível(is)")

            # Tentar inicializar CUDA para verificar se funciona
            try:
                # Verificar se cuDNN está disponível
                try:
                    import paddle.fluid as fluid

                    print(f"PaddlePaddle version: {paddle.__version__}")
                    print("Testando inicialização CUDA...")
                except Exception as import_err:
                    print(f"⚠️  Erro ao importar Paddle Fluid: {import_err}")

                paddle.device.set_device("gpu:0")

                # Criar um tensor pequeno para testar CUDA
                test_tensor = paddle.ones([1, 1])
                result = paddle.sum(test_tensor)

                print(f"✅ CUDA completamente funcional - usando GPU")
                print(f"   Teste tensor executado com sucesso: {result}")
                return True

            except Exception as cuda_err:
                error_msg = str(cuda_err)
                print(f"❌ GPU detectada mas CUDA/cuDNN não funcional!")
                print(f"   Erro: {error_msg}")

                # Verificar se é erro de cuDNN
                if "cudnn" in error_msg.lower():
                    print(f"   PROBLEMA: cuDNN não está configurado corretamente")
                    print(
                        f"   SOLUÇÃO: Use Dockerfile.gpu-fixed que instala cuDNN corretamente"
                    )

                print(f"⚠️  FALLBACK: Usando CPU (modo seguro)")
                print(f"   NOTA: Para usar GPU, corrija a instalação do cuDNN")

                # CRITICAL: Forçar uso de CPU para evitar segfault
                paddle.device.set_device("cpu")
                return False
        else:
            print("⚠️  Nenhuma GPU detectada - usando CPU")
            return False

    except Exception as e:
        print(f"❌ Erro crítico ao detectar GPU: {type(e).__name__}: {e}")
        print(f"⚠️  FALLBACK: Usando CPU (modo seguro)")

        # CRITICAL: Garantir que está em modo CPU
        try:
            import paddle

            paddle.device.set_device("cpu")
        except:
            pass

        return False


USE_GPU = check_gpu_available()

# OTIMIZAÇÃO GPU vs CPU:
# - GPU: 2 instâncias (compartilhadas entre workers) + processamento SERIAL das rotações
# - CPU: 4 instâncias + processamento PARALELO das rotações (racing)
OCR_POOL_SIZE = 2 if USE_GPU else 4

# Criar pool de instâncias PaddleOCR com parâmetros otimizados
print(f"Inicializando pool de {OCR_POOL_SIZE} instâncias PaddleOCR...")
ocr_pool = queue.Queue(maxsize=OCR_POOL_SIZE)


def create_ocr_instance(use_gpu: bool, instance_id: int):
    """Cria instância PaddleOCR com configuração otimizada"""
    # Configuração base
    ocr_config = {
        "use_gpu": use_gpu,
        "use_angle_cls": False,  # Classificador de ângulo desabilitado
        "lang": "pt",  # Português
        # PARÂMETROS OTIMIZADOS PARA CNH/RG
        "det_db_thresh": 0.2,  # Threshold REDUZIDO para detectar texto de baixo contraste
        "det_db_box_thresh": 0.5,  # Threshold de confiança REDUZIDO
        "det_limit_side_len": 1920,  # Preservar detalhes
        "rec_batch_num": (
            6 if use_gpu else 8
        ),  # Batch menor em GPU para economizar memória
        "drop_score": 0.4,  # Aceitar mais reconhecimentos
    }

    # Configuração específica GPU
    if use_gpu:
        ocr_config["gpu_mem"] = 4000  # 4GB por instância (seguro para GPUs com 8GB+)
        ocr_config["enable_mkldnn"] = False  # Desabilitar MKL-DNN em GPU
        ocr_config["gpu_id"] = 0  # Sempre usar GPU 0
    else:
        # Otimizações CPU
        ocr_config["enable_mkldnn"] = True  # Intel MKL-DNN aceleração
        ocr_config["cpu_threads"] = 4

    try:
        instance = PaddleOCR(**ocr_config)
        return instance
    except Exception as e:
        print(f"    ❌ Erro ao criar instância: {e}")
        # Se falhar com GPU, tentar CPU como fallback
        if use_gpu:
            print(f"    🔄 Tentando fallback para CPU...")
            ocr_config["use_gpu"] = False
            ocr_config["enable_mkldnn"] = True
            ocr_config["cpu_threads"] = 4
            if "gpu_mem" in ocr_config:
                del ocr_config["gpu_mem"]
            if "gpu_id" in ocr_config:
                del ocr_config["gpu_id"]
            instance = PaddleOCR(**ocr_config)
            return instance
        else:
            raise


instances_using_gpu = 0
for i in range(OCR_POOL_SIZE):
    print(f"  Criando instância OCR {i+1}/{OCR_POOL_SIZE}...")
    try:
        ocr_instance = create_ocr_instance(USE_GPU, i)
        ocr_pool.put(ocr_instance)
        if USE_GPU:
            instances_using_gpu += 1
        print(f"  ✅ Instância {i+1} criada")
    except Exception as e:
        print(f"  ❌ ERRO CRÍTICO ao criar instância {i+1}: {e}")
        import traceback

        traceback.print_exc()

# Atualizar USE_GPU baseado no que realmente foi criado
if USE_GPU and instances_using_gpu == 0:
    print(f"⚠️  GPU solicitada mas nenhuma instância GPU criada - usando CPU")
    USE_GPU = False

print(
    f"✅ Pool de OCR inicializado: {ocr_pool.qsize()} instâncias ({'GPU - modo SERIAL' if USE_GPU else 'CPU - modo PARALELO'})"
)


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
    import time

    try:
        t0 = time.time()
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        download_time = time.time() - t0
        size_kb = len(response.content) / 1024
        print(f"⏱️  Download da imagem: {download_time:.2f}s ({size_kb:.1f} KB)")
        return response.content
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erro ao baixar imagem {url}: {str(e)}"
        )


def is_pdf(content: bytes) -> bool:
    """Verifica se o conteúdo é um arquivo PDF"""
    return content.startswith(b"%PDF")


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


def preprocess_image_for_ocr(
    img_array: np.ndarray, aggressive: bool = False, keep_color: bool = True
) -> np.ndarray:
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

        print(
            f"PDF convertido: {len(images)} página(s) em 288 DPI (sem pré-processamento)"
        )

    except Exception as e:
        print(f"Erro ao converter PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar PDF: {str(e)}")

    finally:
        # Limpar arquivo temporário
        if tmp_pdf and os.path.exists(tmp_pdf):
            os.unlink(tmp_pdf)

    return images


def preprocess_image_advanced(img: np.ndarray) -> np.ndarray:
    """
    Pré-processamento avançado para melhorar OCR em documentos de baixa qualidade
    Técnicas aplicadas:
    1. Grayscale
    2. Denoise (fastNlMeansDenoising)
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    4. Binarização adaptativa
    5. Morphological operations (opcional)
    """
    try:
        # 1. Converter para grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 2. Denoise - Remove ruído preservando bordas
        denoised = cv2.fastNlMeansDenoising(
            gray, None, h=10, templateWindowSize=7, searchWindowSize=21
        )

        # 3. CLAHE - Melhora contraste local adaptivamente
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 4. Binarização adaptativa (Otsu)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Morphological operations - Remove ruído pequeno
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    except Exception as e:
        print(f"Erro no pré-processamento avançado: {e}")
        # Fallback: retornar grayscale simples
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img


def process_single_rotation_paddle(
    img_array: np.ndarray, angle: int, enhance: bool = True
) -> dict:
    """
    Processa uma única rotação da imagem com PaddleOCR

    Args:
        img_array: Imagem em numpy array
        angle: Ângulo de rotação (0, 90, 180, 270)
        enhance: Se True, aplica pré-processamento avançado (HABILITADO por padrão)

    Returns:
        Dict com angle, text, char_count, num_boxes
    """
    tmp_path = None
    try:
        # Rotacionar imagem usando cv2
        if angle == 90:
            rotated = cv2.rotate(
                img_array, cv2.ROTATE_90_COUNTERCLOCKWISE
            )  # Anti-horário
        elif angle == 180:
            rotated = cv2.rotate(img_array, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)  # Horário
        else:  # 0
            rotated = img_array

        # PRÉ-PROCESSAMENTO AVANÇADO (HABILITADO POR PADRÃO)
        # CLAHE + Denoise + Binarização para melhorar detecção
        if enhance:
            print(f"    [Rot {angle}°] Aplicando pré-processamento avançado...")
            processed = preprocess_image_advanced(rotated)
        else:
            processed = rotated

        # Salvar imagem processada temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Processar com PaddleOCR usando pool (thread-safe)
        with OCRPoolContext() as ocr:
            results = ocr.ocr(tmp_path, cls=False)

        # Extrair texto
        all_text = []
        num_boxes = 0
        if results and results[0]:
            num_boxes = len(results[0])
            for line in results[0]:
                if line and len(line) > 1:
                    text = line[1][0]
                    confidence = line[1][1]
                    all_text.append(text)

        extracted_text = " ".join(all_text)
        char_count = len(extracted_text.strip())

        # Log detalhado incluindo número de bounding boxes
        print(f"  Rotação {angle:4}°: {num_boxes:2} boxes, {char_count:4} chars")

        return {
            "angle": angle,
            "text": extracted_text,
            "char_count": char_count,
            "num_boxes": num_boxes,
        }

    except Exception as e:
        print(f"Erro ao processar rotação {angle}°: {e}")
        return {"angle": angle, "text": "", "char_count": 0, "num_boxes": 0}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Função detect_image_orientation removida - usando apenas PaddleOCR racing


def perform_ocr(
    image_bytes: bytes, enhance: bool = True, expected_name: str = None
) -> str:
    """
    Realiza OCR em uma imagem (bytes) e retorna texto concatenado

    ESTRATÉGIA OTIMIZADA (APENAS PADDLEOCR RACING):
    1. Pré-processamento UMA VEZ antes do racing (CLAHE + Denoise + Binarização)
    2. Racing PARALELO das 4 rotações (0°, 90°, 180°, 270°)
    3. Escolhe rotação com MAIS bounding boxes (prioridade) + mais texto
    4. Retorna resultado do PaddleOCR (sem fallback)

    Performance: ~8-12s por imagem (RÁPIDO)

    Args:
        image_bytes: Bytes da imagem
        enhance: Se True, aplica pré-processamento avançado (HABILITADO por padrão)
        expected_name: Nome esperado no documento (opcional, não usado atualmente)
    """
    import time

    start_time = time.time()

    try:
        # Converter bytes para numpy array usando cv2
        t0 = time.time()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(f"⏱️  Decodificação da imagem: {time.time() - t0:.2f}s")

        if img is None:
            print("Erro: não foi possível decodificar a imagem")
            return ""

        # Redimensionar se imagem for muito grande (> 1920px para preservar detalhes)
        height, width = img.shape[:2]
        max_dimension = (
            1920  # Aumentado de 2000 para 1920 (compatível com det_limit_side_len)
        )
        if max(height, width) > max_dimension:
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            img = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4
            )
            print(
                f"Imagem redimensionada: {width}x{height} → {new_width}x{new_height}px"
            )
        else:
            print(f"Imagem original: {width}x{height}px")

        # PRÉ-PROCESSAMENTO UMA VEZ (antes do racing)
        # Aplicar CLAHE + Denoise + Binarização para melhorar detecção
        if enhance:
            t1 = time.time()
            print("Aplicando pré-processamento avançado na imagem...")
            img = preprocess_image_advanced(img)
            print(f"⏱️  Pré-processamento concluído em {time.time() - t1:.2f}s")

        # MODO ADAPTATIVO: SERIAL (GPU) vs PARALELO (CPU)
        t2 = time.time()
        rotations = [0, 90, 180, 270]
        results = []

        if USE_GPU:
            # GPU: Processar rotações em SÉRIE (uma por vez) para evitar sobrecarga de memória
            print("Testando múltiplas rotações em SÉRIE (modo GPU)...")
            for angle in rotations:
                # Desabilitar enhance pois já preprocessamos
                result = process_single_rotation_paddle(img, angle, False)
                results.append(result)
        else:
            # CPU: Processar rotações em PARALELO (racing - mais rápido)
            print("Testando múltiplas rotações em PARALELO (racing - modo CPU)...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Desabilitar enhance pois já preprocessamos
                futures = {
                    executor.submit(
                        process_single_rotation_paddle, img, angle, False
                    ): angle
                    for angle in rotations
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

        mode = "SERIAL (GPU)" if USE_GPU else "PARALELO (CPU)"
        print(
            f"⏱️  Processamento de rotações ({mode}) concluído em {time.time() - t2:.2f}s"
        )

        # SELEÇÃO INTELIGENTE: Priorizar num_boxes (indicador de melhor detecção)
        # Ordenar por: 1) num_boxes (PRINCIPAL), 2) char_count (secundário)
        best_result = max(results, key=lambda x: (x["num_boxes"], x["char_count"]))
        best_angle = best_result["angle"]
        best_text = best_result["text"]
        best_chars = best_result["char_count"]
        best_boxes = best_result["num_boxes"]

        # Log detalhado de todas as rotações testadas
        print(f"\nResultado do racing de rotações:")
        for r in sorted(
            results, key=lambda x: (x["num_boxes"], x["char_count"]), reverse=True
        ):
            marker = "✓ MELHOR" if r["angle"] == best_angle else "    "
            num_boxes = r.get("num_boxes", 0)
            print(
                f"  {marker} {r['angle']:3}°: {num_boxes:2} boxes, {r['char_count']:4} caracteres"
            )

        print(
            f"Usando rotação {best_angle}° ({best_boxes} boxes, {best_chars} caracteres extraídos)"
        )

        # Retornar resultado do PaddleOCR (sem fallback)
        print(f"⏱️  TEMPO TOTAL: {time.time() - start_time:.2f}s")
        return best_text

    except Exception as e:
        print(f"Erro no OCR: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return ""


# ============================================================================
# FUNÇÕES DE VALIDAÇÃO DE NOME COM FUZZY MATCHING
# ============================================================================


def normalize_text(text: str) -> str:
    """
    Normaliza texto para comparação
    - Remove acentos
    - Converte para minúsculas
    - Remove pontuação, vírgulas, espaços
    - Remove quebras de linha
    """
    if not text:
        return ""

    import unicodedata

    # Normalização NFD (decompõe caracteres acentuados)
    text = unicodedata.normalize("NFD", text)

    # Remove marcas diacríticas (acentos)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")

    # Converte para minúsculas
    text = text.lower()

    # Remove pontuação e caracteres especiais
    text = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()]", "", text)

    # Remove todos os espaços em branco
    text = re.sub(r"\s+", "", text)

    # Remove quebras de linha
    text = text.replace("\n", "").replace("\r", "")

    return text.strip()


def is_name_in_text(expected_name: str, ocr_text: str) -> bool:
    """
    Verifica se o nome esperado está presente no texto OCR (validação simples por substring)

    Args:
        expected_name: Nome completo esperado (ex: "jose benedito souza da hora")
        ocr_text: Texto extraído pelo OCR

    Returns:
        True se o nome normalizado está contido no texto normalizado, False caso contrário
    """
    if not expected_name or not ocr_text:
        return False

    # Normaliza ambos os textos
    normalized_name = normalize_text(expected_name)
    normalized_text = normalize_text(ocr_text)

    # Verifica se o nome está contido no texto (substring)
    found = normalized_name in normalized_text

    print(f"\n{'='*80}")
    print(f"VALIDAÇÃO DE NOME (SUBSTRING SIMPLES)")
    print(f"{'='*80}")
    print(f'Nome esperado: "{expected_name}"')
    print(f'Nome normalizado: "{normalized_name}"')
    print(f'Texto normalizado (primeiros 200 chars): "{normalized_text[:200]}..."')
    print(f"Nome encontrado no texto: {'SIM ✅' if found else 'NÃO ❌'}")
    print(f"{'='*80}\n")

    return found


def levenshtein_distance(str1: str, str2: str) -> int:
    """
    Calcula a distância de Levenshtein entre duas strings
    Mede o número mínimo de edições (inserções, deleções, substituições)
    necessárias para transformar uma string em outra
    """
    len1 = len(str1)
    len2 = len(str2)

    # Cria matriz de distâncias
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    # Inicializa primeira coluna e linha
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    # Calcula distâncias
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # Deleção
                matrix[i][j - 1] + 1,  # Inserção
                matrix[i - 1][j - 1] + cost,  # Substituição
            )

    return matrix[len1][len2]


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calcula similaridade percentual entre duas strings
    Baseado na distância de Levenshtein
    Retorna: 0 a 1 (0 = totalmente diferente, 1 = idêntico)
    """
    if str1 == str2:
        return 1.0
    if not str1 or not str2:
        return 0.0

    distance = levenshtein_distance(str1, str2)
    max_length = max(len(str1), len(str2))

    return 1.0 - (distance / max_length)


def find_best_word_match(
    word: str, text: str, min_similarity: float = 0.6, debug: bool = False
) -> dict:
    """
    Busca a melhor correspondência de uma palavra no texto usando janela deslizante

    Returns:
        dict: {found: bool, similarity: float, matched_text: str, position: int}
    """
    if not word or not text or len(word) < 2:
        return {"found": False, "similarity": 0.0, "matched_text": "", "position": -1}

    best_similarity = 0.0
    best_match = ""
    best_position = -1

    # PRIMEIRO: Busca exata (substring)
    if word in text:
        position = text.find(word)
        if debug:
            print(f'      🎯 MATCH EXATO encontrado: "{word}" na posição {position}')
        return {
            "found": True,
            "similarity": 1.0,
            "matched_text": word,
            "position": position,
        }

    # SEGUNDO: Testa com palavras de tamanho variável (±50% do tamanho esperado)
    min_length = max(2, int(len(word) * 0.5))
    max_length = int(len(word) * 1.5)

    if debug:
        print(f'      Buscando "{word}" (tamanho: {len(word)})')
        print(f"      Janela: {min_length} a {max_length} caracteres")
        print(f"      Texto tem {len(text)} caracteres")

    checks_count = 0
    max_checks = 20000  # Limite para performance

    for length in range(min_length, max_length + 1):
        if checks_count >= max_checks:
            break
        for i in range(len(text) - length + 1):
            if checks_count >= max_checks:
                break
            checks_count += 1

            chunk = text[i : i + length]
            similarity = calculate_similarity(word, chunk)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = chunk
                best_position = i

                if debug and similarity >= min_similarity:
                    print(f'      ✓ Candidato: "{chunk}" ({similarity*100:.1f}%)')

    if debug:
        status = "ACEITO ✅" if best_similarity >= min_similarity else "REJEITADO ❌"
        print(
            f'      Melhor match: "{best_match}" ({best_similarity*100:.1f}%) - {status}'
        )

    return {
        "found": best_similarity >= min_similarity,
        "similarity": best_similarity,
        "matched_text": best_match,
        "position": best_position,
    }


def find_name_in_text(
    expected_name: str, ocr_text: str, threshold: float = 0.70
) -> dict:
    """
    Verifica se um nome está presente no texto usando técnicas avançadas

    Estratégia:
    1. Match exato (normalizado)
    2. Busca palavra por palavra com fuzzy matching
    3. Aceita se encontrar ≥70% das palavras com similaridade ≥60%
    4. Calcula score ponderado considerando todas as palavras
    5. Primeiro nome tem peso 4x maior (essencial!)

    Returns:
        dict: {found: bool, confidence: float, matched_text: str, method: str, details: list, stats: dict}
    """
    if not expected_name or not ocr_text:
        return {
            "found": False,
            "confidence": 0.0,
            "matched_text": "",
            "method": "empty_input",
            "details": [],
        }

    # Normaliza ambos os textos
    normalized_name = normalize_text(expected_name)
    normalized_ocr_text = normalize_text(ocr_text)

    print(f"\n{'='*80}")
    print("🔍 ANÁLISE DE COMPARAÇÃO DE TEXTO (MODO INTELIGENTE + DEBUG)")
    print(f"{'='*80}")
    print(f'Nome esperado: "{expected_name}"')
    print(f'Nome normalizado: "{normalized_name}"')
    print(f"Threshold global: {threshold}")
    print(f"\n📝 TEXTO OCR ORIGINAL (primeiros 500 caracteres):")
    print(f"\"{ocr_text[:500]}{'...' if len(ocr_text) > 500 else ''}\"")
    print(f"\n📝 TEXTO OCR NORMALIZADO (primeiros 500 caracteres):")
    print(
        f"\"{normalized_ocr_text[:500]}{'...' if len(normalized_ocr_text) > 500 else ''}\""
    )
    print(f"Tamanho total do texto normalizado: {len(normalized_ocr_text)} caracteres")
    print(f"{'='*80}\n")

    # 1. Tentativa: Match exato (texto normalizado)
    if normalized_name in normalized_ocr_text:
        print(f"✅ MATCH EXATO encontrado!")
        print(f"   Método: substring exata")
        print(f"   Confiança: 1.0\n")
        return {
            "found": True,
            "confidence": 1.0,
            "matched_text": normalized_name,
            "method": "exact_match",
            "details": [
                {"word": normalized_name, "similarity": 1.0, "matched": normalized_name}
            ],
        }

    # 2. Análise palavra por palavra com fuzzy matching INTELIGENTE
    name_words = [w for w in expected_name.split() if len(w) > 1]
    print(f"📊 Análise palavra por palavra ({len(name_words)} palavras):\n")

    word_results = []
    total_weighted_score = 0.0
    total_weight = 0.0
    first_name_score = 0.0

    for i, original_word in enumerate(name_words):
        normalized_word = normalize_text(original_word)

        # Pula palavras muito pequenas
        if len(normalized_word) < 2:
            print(f'   ⏭️  "{original_word}" - Palavra muito curta, ignorada')
            continue

        # Busca a melhor correspondência desta palavra no texto
        word_match = find_best_word_match(
            normalized_word, normalized_ocr_text, 0.50, debug=True
        )

        # PESO ESPECIAL: Primeiro nome tem peso 4x maior (essencial!)
        is_first_name = i == 0
        weight = len(normalized_word)

        if is_first_name:
            weight = weight * 4  # Primeiro nome é 4x mais importante!
            first_name_score = word_match["similarity"]

        total_weight += weight

        word_score = word_match["similarity"]
        total_weighted_score += word_score * weight

        word_results.append(
            {
                "original": original_word,
                "normalized": normalized_word,
                "similarity": word_score,
                "matched": word_match["matched_text"],
                "found": word_match["found"],
                "weight": weight,
                "is_first_name": is_first_name,
            }
        )

        icon = "✅" if word_match["found"] else "❌"
        confidence = word_score * 100
        first_name_tag = " 🌟 PRIMEIRO NOME (PESO 4x)" if is_first_name else ""
        print(f'   {icon} "{original_word}" ({normalized_word}){first_name_tag}')
        print(f"      Similaridade: {confidence:.1f}%")
        print(f"      Match: \"{word_match['matched_text'] or 'não encontrado'}\"")
        print(f"      Peso: {weight}")
        print()

    # Calcula score final ponderado
    final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0

    # Conta quantas palavras foram encontradas com similaridade >= 0.60
    words_found_60 = len([w for w in word_results if w["similarity"] >= 0.60])
    percentage_words_found = words_found_60 / len(word_results) if word_results else 0.0

    # Separa primeiro nome das outras palavras
    first_name_result = next((w for w in word_results if w["is_first_name"]), None)
    other_words = [w for w in word_results if not w["is_first_name"]]
    other_words_found = len([w for w in other_words if w["similarity"] >= 0.60])
    percentage_other_words = (
        other_words_found / len(other_words) if other_words else 0.0
    )

    print(f"\n{'─'*80}")
    print(f"📈 SCORE FINAL:")
    print(f"   Score ponderado: {final_score*100:.1f}%")
    print(
        f"   Palavras encontradas (≥60%): {words_found_60}/{len(word_results)} ({percentage_words_found*100:.1f}%)"
    )
    print(
        f"   🌟 Primeiro nome: {first_name_score*100:.1f}% {'✅ ENCONTRADO' if first_name_score >= 0.70 else '❌ NÃO ENCONTRADO'}"
    )
    print(
        f"   📝 Outras palavras: {other_words_found}/{len(other_words)} ({percentage_other_words*100:.1f}%)"
    )
    print(f"   Threshold necessário: {threshold*100:.1f}%")
    print(f"{'─'*80}\n")

    # Estratégia de decisão AVANÇADA
    first_name_found = first_name_score >= 0.70
    has_other_words = percentage_other_words >= 0.50

    criterion1 = final_score >= threshold
    criterion2 = percentage_words_found >= 0.70
    criterion3 = first_name_found and has_other_words

    found = criterion1 or criterion2 or criterion3

    matched_words = " ".join([w["matched"] for w in word_results if w["found"]])

    if found:
        print(f"✅ NOME ENCONTRADO!")
        print(f"   Método: fuzzy_intelligent")
        print(f"   Confiança: {final_score*100:.1f}%")

        criteria = []
        if criterion1:
            criteria.append(
                f"✓ score ponderado ({final_score*100:.1f}% ≥ {threshold*100:.0f}%)"
            )
        if criterion2:
            criteria.append(
                f"✓ palavras encontradas ({percentage_words_found*100:.0f}% ≥ 70%)"
            )
        if criterion3:
            criteria.append(
                f"✓ 🌟 primeiro nome ({first_name_score*100:.1f}% ≥ 70%) + outras palavras ({percentage_other_words*100:.0f}% ≥ 50%)"
            )
        print(
            f"   Critério(s) atendido(s):\n      {chr(10).join(['      ' + c for c in criteria])}"
        )
        print(f'   Match: "{matched_words}"\n')
    else:
        print(f"❌ NOME NÃO ENCONTRADO")
        print(f"   Razões:")
        if not first_name_found:
            print(
                f"      • Primeiro nome NÃO encontrado ({first_name_score*100:.1f}% < 70%) ❌"
            )
        elif not has_other_words:
            print(
                f"      • Primeiro nome encontrado MAS poucas outras palavras ({percentage_other_words*100:.0f}% < 50%) ❌"
            )
        if final_score < threshold:
            print(
                f"      • Score ponderado baixo ({final_score*100:.1f}% < {threshold*100:.0f}%) ❌"
            )
        if percentage_words_found < 0.70:
            print(
                f"      • Poucas palavras encontradas ({percentage_words_found*100:.0f}% < 70%) ❌"
            )
        print()

    return {
        "found": found,
        "confidence": final_score,
        "matched_text": matched_words,
        "method": "fuzzy_intelligent",
        "details": word_results,
        "stats": {
            "total_words": len(word_results),
            "words_found": words_found_60,
            "percentage_found": percentage_words_found,
            "weighted_score": final_score,
            "first_name_score": first_name_score,
            "first_name_found": first_name_found,
            "other_words_total": len(other_words),
            "other_words_found": other_words_found,
            "other_words_percentage": percentage_other_words,
        },
    }


def extract_cpf(text: str) -> Optional[str]:
    """Extrai CPF do texto"""
    # Padrões: 123.456.789-01 ou 12345678901
    patterns = [
        r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
        r"\b\d{2}\.\d{3}\.\d{3}-\d{2}\b",
        r"\b\d{11}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            cpf = match.group()
            # Verificar se tem 11 dígitos
            nums = re.sub(r"\D", "", cpf)
            if len(nums) == 11:
                return cpf
    return None


def extract_rg(text: str) -> Optional[str]:
    """Extrai RG do texto"""
    # Padrões comuns: 12.345.678-9, 12345678-9, 08-055535083
    patterns = [
        r"\b\d{2}\.\d{3}\.\d{3}-\d{1,2}\b",
        r"\b\d{8,9}-\d{1,2}\b",
        r"\b\d{2}-\d{9}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    return None


def extract_dates(text: str) -> Dict[str, Optional[str]]:
    """Extrai datas do texto (nascimento e expedição)"""
    dates = {"data_nascimento": None, "data_expedicao": None}

    # Padrões de data: DD/MM/YYYY, DD-MM-YYYY, DD=MM=YYYY, DDMMYYYY
    date_patterns = [r"\b\d{2}[-/=\.]\d{2}[-/=\.]\d{4}\b", r"\b\d{8}\b"]

    # Procurar "DATA DE NASCIMENTO" ou "NASC" seguido de data
    nasc_match = re.search(
        r"(?:DATA.*?NASC|NASC|DAT).*?(\d{2}[-/=\.]\d{2}[-/=\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if nasc_match:
        dates["data_nascimento"] = nasc_match.group(1)

    # Procurar "DATA DE EXPEDIÇÃO" ou "EXPEDICAO" seguido de data
    exp_match = re.search(
        r"(?:DATA.*?EXPEDI|EXPEDI[CÇ]).*?(\d{2}[-/=\.]\d{2}[-/=\.]\d{4})",
        text,
        re.IGNORECASE,
    )
    if exp_match:
        dates["data_expedicao"] = exp_match.group(1)

    # Se não encontrou com contexto, pegar as duas últimas datas do documento
    all_dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Converter para formato padrão
            if "-" in match or "/" in match or "=" in match or "." in match:
                all_dates.append(match)
            elif len(match) == 8:  # DDMMYYYY
                formatted = f"{match[:2]}/{match[2:4]}/{match[4:]}"
                all_dates.append(formatted)

    # Se encontrou datas mas não identificou contexto
    if all_dates:
        if not dates["data_nascimento"] and len(all_dates) >= 1:
            dates["data_nascimento"] = all_dates[0]
        if not dates["data_expedicao"] and len(all_dates) >= 2:
            dates["data_expedicao"] = all_dates[-1]

    return dates


def extract_names(text: str) -> Dict[str, Optional[str]]:
    """Extrai nomes (titular, mãe, pai) do texto"""
    names = {"nome": None, "mae": None, "pai": None}

    # Palavras comuns que não são nomes
    stopwords = {
        "DE",
        "DA",
        "DO",
        "DOS",
        "DAS",
        "E",
        "O",
        "A",
        "OS",
        "AS",
        "REPUBLICA",
        "FEDERATIVA",
        "BRASIL",
        "ESTADO",
        "CARTEIRA",
        "IDENTIDADE",
        "RG",
        "CPF",
        "DATA",
        "NASCIMENTO",
        "EXPEDICAO",
        "ASSINATURA",
        "DIRETOR",
        "DIRETORA",
        "LEI",
        "VALIDA",
        "TODO",
        "TERRITORIO",
        "NACIONAL",
        "SECRETARIA",
        "SEGURANCA",
        "PUBLICA",
        "POLICIA",
        "DELEGADO",
        "SSP",
        "RGD",
        "NOME",
        "FILIACAO",
        "MAE",
        "PAI",
        "REGISTRO",
        "GERAL",
        "DOC",
        "ORIGEM",
        "NATURALIDADE",
    }

    # Dividir texto em palavras
    lines = text.split()

    # Procurar padrões de nomes (sequências de palavras em maiúsculas)
    potential_names = []
    i = 0
    while i < len(lines):
        word = lines[i]
        # Se encontrar palavra em maiúsculas (possível nome)
        if (
            word.isupper()
            and len(word) > 2
            and word.isalpha()
            and word not in stopwords
        ):
            name_parts = [word]
            # Continuar pegando palavras em maiúsculas
            j = i + 1
            while j < len(lines) and j < i + 6:  # Limite de 6 palavras para um nome
                next_word = lines[j]
                if next_word.isupper() and (
                    next_word.isalpha() or next_word in {"DE", "DA", "DO", "DOS", "DAS"}
                ):
                    if len(next_word) > 1:
                        name_parts.append(next_word)
                    j += 1
                else:
                    break

            # Se encontrou pelo menos 2 palavras (ignorando preposições), pode ser um nome
            actual_words = [
                w for w in name_parts if w not in {"DE", "DA", "DO", "DOS", "DAS"}
            ]
            if len(actual_words) >= 2:
                full_name = " ".join(name_parts)
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
        names["nome"] = valid_names[-1]
        # Penúltimo e antepenúltimo são mãe e pai
        names["mae"] = valid_names[-3]
        names["pai"] = valid_names[-2]
    elif len(valid_names) >= 2:
        names["nome"] = valid_names[-1]
        names["mae"] = valid_names[-2]
    elif len(valid_names) >= 1:
        names["nome"] = valid_names[-1]

    return names


def extract_location(text: str) -> Optional[str]:
    """Extrai localização/estado do texto"""
    # Estados brasileiros (siglas e nomes)
    estados = [
        "AC",
        "AL",
        "AP",
        "AM",
        "BA",
        "CE",
        "DF",
        "ES",
        "GO",
        "MA",
        "MT",
        "MS",
        "MG",
        "PA",
        "PB",
        "PR",
        "PE",
        "PI",
        "RJ",
        "RN",
        "RS",
        "RO",
        "RR",
        "SC",
        "SP",
        "SE",
        "TO",
    ]

    for estado in estados:
        if re.search(rf"\b{estado}\b", text):
            # Tentar pegar cidade também
            city_match = re.search(rf"(\w+(?:\s+\w+)?)\s+{estado}", text)
            if city_match:
                return f"{city_match.group(1)} {estado}"
            return estado

    return None


def extract_document_type(text: str) -> Optional[str]:
    """Identifica tipo de documento"""
    text_upper = text.upper()

    if "CARTEIRA DE IDENTIDADE" in text_upper or "RG" in text_upper:
        return "RG - Carteira de Identidade"
    elif (
        "CNH" in text_upper
        or "HABILITACAO" in text_upper
        or "DRIVER LICENSE" in text_upper
    ):
        return "CNH - Carteira Nacional de Habilitação"
    elif "CTPS" in text_upper or "TRABALHO" in text_upper:
        return "CTPS - Carteira de Trabalho"

    return "Documento de Identificação"


def extract_fields_from_ocr(text: str) -> Dict[str, Any]:
    """Extrai campos estruturados do texto do OCR"""
    fields = {
        "documento_tipo": extract_document_type(text),
        "cpf": extract_cpf(text),
        "rg": extract_rg(text),
        "local": extract_location(text),
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
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
@app.post("/health")
async def health():
    """Health check endpoint (GET e POST) para SaladCloud"""
    return {"status": "ok", "service": "paddleocr-api"}


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
        lines = [
            {"text": line, "score": 1.0} for line in text.split("\n") if line.strip()
        ]
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
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]

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
                combined_text = " ".join(all_text)
                extracted_fields = extract_fields_from_ocr(combined_text)

                return {
                    "ocrText": combined_text,
                    "extractedFields": extracted_fields,
                    "lines": lines,
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
                print(
                    f"PDF {index+1}: texto nativo extraído ({len(native_text)} caracteres)"
                )

            # Converter PDF em imagens para OCR
            page_images = convert_pdf_to_images(content_bytes)
            print(f"PDF {index+1}: {len(page_images)} página(s) convertidas para OCR")

            return (index, {"images": page_images, "native_text": native_text})
        else:
            # Imagem normal - retornar como lista
            return (index, {"images": [content_bytes], "native_text": ""})
    except Exception as e:
        print(f"Erro ao baixar/preparar URL {url}: {e}")
        return (index, {"images": [], "native_text": ""})


async def process_single_doc_async(
    index: int, doc_data: dict, expected_name: str = None
) -> tuple:
    """
    Processa um único documento de forma assíncrona
    doc_data = {'images': [bytes...], 'native_text': str}
    expected_name: Nome esperado no documento (opcional, para validação de fallback)
    Retorna texto combinado: texto nativo + OCR
    """
    try:
        image_list = doc_data.get("images", [])
        native_text = doc_data.get("native_text", "")

        # Processar OCR das imagens
        print(f"Fazendo OCR do documento {index+1}...")
        ocr_texts = []

        for page_num, image_bytes in enumerate(image_list, 1):
            if len(image_list) > 1:
                print(f"  Página {page_num}/{len(image_list)}...")

            # Executar OCR em thread separada (perform_ocr é bloqueante)
            loop = asyncio.get_event_loop()
            # Passar expected_name para perform_ocr
            page_text = await loop.run_in_executor(
                None,
                lambda img=image_bytes, name=expected_name: perform_ocr(
                    img, expected_name=name
                ),
            )
            if page_text:
                ocr_texts.append(page_text)

        ocr_combined = " ".join(ocr_texts)

        # Combinar texto nativo + OCR
        all_texts = []
        if native_text:
            all_texts.append(native_text)
            print(f"Documento {index+1}: texto nativo ({len(native_text)} chars)")
        if ocr_combined:
            all_texts.append(ocr_combined)
            print(f"Documento {index+1}: OCR ({len(ocr_combined)} chars)")

        combined = " ".join(all_texts)
        if combined:
            print(
                f"Documento {index+1} concluído: total {len(combined)} caracteres (nativo + OCR)"
            )
        else:
            print(f"Documento {index+1}: nenhum texto extraído")

        return (index, combined)
    except Exception as e:
        print(f"Erro ao processar documento {index+1}: {e}")
        return (index, "")


async def process_ocr_parallel(
    prepared_images: List[tuple], expected_name: str = None
) -> str:
    """
    Processa OCR de forma PARALELA - múltiplas imagens simultaneamente!
    prepared_images: lista de (index, dict com 'images' e 'native_text')
    expected_name: Nome esperado no documento (opcional, para validação de fallback)
    Combina texto nativo + OCR para cada documento
    """
    print(f"Processando {len(prepared_images)} documentos em PARALELO...")
    print(
        f"DEBUG: Documentos a processar: {[(idx, len(doc['images'])) for idx, doc in prepared_images]}"
    )

    # Criar tasks para processar todos os documentos simultaneamente
    tasks = [
        process_single_doc_async(index, doc_data, expected_name=expected_name)
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

    final_text = " ".join(all_texts)
    print(
        f"DEBUG: Texto final: {len(final_text)} chars (de {len(all_texts)} documentos)"
    )

    return final_text


def process_ocr_sequential(
    prepared_images: List[tuple], expected_name: str = None
) -> str:
    """
    Processa OCR de forma SEQUENCIAL para evitar race conditions
    prepared_images: lista de (index, [image_bytes])
    expected_name: Nome esperado no documento (opcional, para validação de fallback)
    """
    all_texts = []

    for index, image_list in sorted(prepared_images):
        try:
            print(f"Fazendo OCR do documento {index+1}...")
            doc_texts = []

            for page_num, image_bytes in enumerate(image_list, 1):
                if len(image_list) > 1:
                    print(f"  Página {page_num}/{len(image_list)}...")

                page_text = perform_ocr(image_bytes, expected_name=expected_name)
                if page_text:
                    doc_texts.append(page_text)

            combined = " ".join(doc_texts)
            if combined:
                all_texts.append(combined)
                print(f"OCR {index+1} concluído ({len(combined)} caracteres)")
            else:
                print(f"OCR {index+1} não extraiu texto")

        except Exception as e:
            print(f"Erro ao fazer OCR do documento {index+1}: {e}")

    return " ".join(all_texts)


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
            print(
                f"Deduplicação URL: {total_urls} URLs → {unique_url_count} únicas ({duplicates_by_url} duplicadas removidas)"
            )

        # FASE 1: Baixar e preparar imagens/PDFs EM PARALELO
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submete todas as tarefas de download em paralelo
            futures = [
                loop.run_in_executor(
                    executor, download_and_prepare, url, idx, unique_url_count
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
            image_list = doc_data.get("images", [])
            native_text = doc_data.get("native_text", "")
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
                    print(
                        f"Imagem duplicada detectada (hash: {img_hash[:8]}...) - pulando"
                    )

            # Se há imagens únicas neste documento, adicionar à lista
            if unique_images_in_doc:
                deduplicated_docs.append(
                    (idx, {"images": unique_images_in_doc, "native_text": native_text})
                )

        duplicates_by_hash = total_images_downloaded - unique_image_count

        if duplicates_by_hash > 0:
            print(
                f"Deduplicação Hash: {total_images_downloaded} imagens → {unique_image_count} únicas ({duplicates_by_hash} duplicadas removidas)"
            )

        # Usar documentos deduplicados
        unique_images = deduplicated_docs

        # FASE 3: Processar OCR EM PARALELO apenas das imagens únicas
        # Passar nome esperado para validação de fallback
        expected_name = request.nome if hasattr(request, "nome") else None
        combined_text = await process_ocr_parallel(
            unique_images, expected_name=expected_name
        )

        # # Extrair campos estruturados do texto
        # extracted_fields = extract_fields_from_ocr(combined_text)

        return {
            "ocrText": combined_text,
            # "extractedFields": extracted_fields,
            "status": "OK",
            "stats": {
                "total_urls": total_urls,
                "unique_urls": unique_url_count,
                "unique_images": unique_image_count,
                "duplicates_removed": duplicates_by_url + duplicates_by_hash,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")


salad_machine_id = os.getenv("SALAD_MACHINE_ID", "localhost")


@app.get("/hello")
async def hello_world():
    return {"message": "Hello World", "salad_machine_id": salad_machine_id}


@app.get("/started")
async def started():
    return {"message": "Started", "salad_machine_id": salad_machine_id}


@app.get("/ready")
async def ready():
    return {"message": "Ready", "salad_machine_id": salad_machine_id}


@app.get("/live")
async def live():
    return {"message": "Live", "salad_machine_id": salad_machine_id}


@app.get("/health")
async def health():
    return {"message": "Healthy", "salad_machine_id": salad_machine_id}
