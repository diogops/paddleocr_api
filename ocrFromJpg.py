import pytesseract
from PIL import Image, UnidentifiedImageError, ImageEnhance, ImageFilter
import cv2
import numpy as np
import logging

def perform_ocr_on_jpg(image_path: str) -> str:
    """
    Perform OCR on a JPG image with advanced preprocessing.
    
    Args:
        image_path (str): Path to the JPG image file.
        
    Returns:
        str: Extracted text from the image, or an error message.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting OCR on image: {image_path}")
        
        # Open the image
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            logger.error(f"File not found: {image_path}")
            return "file not found"
        except UnidentifiedImageError:
            logger.error(f"Invalid image file: {image_path}")
            return "invalid image file"
        
        # Ensure file format is JPG
        if image.format not in ["JPEG", "JPG"]:
            logger.error(f"Invalid image format: {image.format}. Only JPG images are supported.")
            return "invalid image format"
        
        # Convert image to grayscale
        logger.info("Preprocessing the image (grayscale conversion, contrast enhancement, noise removal)...")
        gray = image.convert("L")

        # Increase contrast
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2.5)  # Aumenta o contraste para melhorar o OCR

        # Apply sharpening filter
        sharpness = ImageEnhance.Sharpness(gray)
        gray = sharpness.enhance(2.0)  # Ajuste para melhorar a nitidez do texto

        # Convert PIL image to OpenCV format for further processing
        open_cv_image = np.array(gray)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(open_cv_image, (5, 5), 0)

        # Apply Otsu's Binarization
        _, thresh = cv2.threshold(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert back to PIL format for OCR
        processed_image = Image.fromarray(thresh)

        # Perform OCR with improved settings
        logger.info("Performing OCR...")
        custom_config = "--psm 3 --oem 3"
        text = pytesseract.image_to_string(processed_image, lang="eng", config=custom_config)
        
        logger.info("OCR completed successfully")
        return text.strip()
    
    except Exception as e:
        logger.error(f"Unexpected error during OCR: {str(e)}", exc_info=True)
        return "unexpected error during ocr"



# import pytesseract
# from PIL import Image, UnidentifiedImageError
# import logging

# def perform_ocr_on_jpg(image_path: str) -> str:
#     """
#     Perform OCR on a JPG image and extract text.
    
#     Args:
#         image_path (str): Path to the JPG image file.
        
#     Returns:
#         str: Extracted text from the image, or an empty string if an error occurs.
#     """
#     # Configure logging
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger(__name__)
    
#     try:
#         logger.info(f"Starting OCR on image: {image_path}")
        
#         # Open the image
#         try:
#             image = Image.open(image_path)
#         except FileNotFoundError:
#             logger.error(f"File not found: {image_path}")
#             return "file not found"
#         except UnidentifiedImageError:
#             logger.error(f"Invalid image file: {image_path}")
#             return "invalid image file"
        
#         # Ensure file format is JPG
#         if image.format not in ["JPEG", "JPG"]:
#             logger.error(f"Invalid image format: {image.format}. Only JPG images are supported.")
#             return "invalid image format"
        
#         # Perform OCR
#         logger.info("Performing OCR...")
#         text = pytesseract.image_to_string(image, lang="eng")
        
#         logger.info("OCR completed successfully")
#         return text.strip()
    
#     except Exception as e:
#         logger.error(f"Unexpected error during OCR: {str(e)}", exc_info=True)
#         return "unexpected error during ocr"
