import os
import fitz  # PyMuPDF
import logging
import uuid

def convert_pdf_to_jpg(pdf_path, output_dir, zoom=2.0):
    """
    Convert PDF file to high-quality JPG images and save in the specified output directory
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory where JPG images will be saved
        zoom (float): Zoom factor for resolution (default 2.0)
        
    Returns:
        list: List of paths to the generated JPG files
    """
    # Configurar logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    jpg_paths = []    

    try:
        logger.info(f"Iniciando conversão do PDF: {pdf_path}")
        
        if not os.path.isfile(pdf_path):
            logger.error(f"Arquivo não encontrado: {pdf_path}")
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Diretório de saída criado: {output_dir}")

        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        logger.info(f"PDF aberto com sucesso. Total de páginas: {total_pages}")
        
        for page_number in range(total_pages):
            logger.info(f"Processando página {page_number + 1} de {total_pages}")
            
            page = pdf_document[page_number]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            random_filename = str(uuid.uuid4())
            jpg_path = os.path.join(output_dir, f"{random_filename}.jpg")
            
            logger.info(f"Salvando página {page_number + 1} como JPG em {jpg_path}")
            pix.save(jpg_path)

            jpg_paths.append(jpg_path)
            logger.info(f"Página {page_number + 1} convertida com sucesso: {jpg_path}")

        logger.info(f"Processo finalizado. Total de imagens geradas: {len(jpg_paths)}")
        return jpg_paths

    except Exception as e:
        logger.error(f"Erro ao converter PDF para JPG: {str(e)}", exc_info=True)
        return []

# Exemplo de uso para chamada a partir do Node.js
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Uso: python script.py <caminho_pdf> <diretorio_saida>")
    else:
        pdf_path = sys.argv[1]
        output_dir = sys.argv[2]
        jpg_paths = convert_pdf_to_jpg(pdf_path, output_dir)
        print("\n".join(jpg_paths))
