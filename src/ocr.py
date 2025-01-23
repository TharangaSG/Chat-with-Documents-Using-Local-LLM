import io
import logging
from typing import Optional
import pytesseract
from PIL import Image
from PyPDF2 import PageObject, PdfReader
from src.constants import LOG_FILE_PATH
from src.utils import clean_text, setup_logging


#configure logging
setup_logging()
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file_path: str) -> str:
    """
    Extracts text from a PDF file using PyPDF2 and Tesseract OCR.

    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF file.
    """
    text = ""
    with open(pdf_file_path, 'rb') as f:
        pdf_reader = PdfReader(f)  
        logger.info(f"opened pdf file {pdf_file_path}")

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            try:
                page_text = page.extractText()
                if page_text:
                    text += page_text
                    logger.info(f"extracted text from page {page_num} without OCR")
                else:
                    logger.info(f"No text found on page {page_num}; attempting OCR.")
                    text += extract_text_from_images(page)
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}") 
    
    cleaned_text = clean_text(text)
    logger.info(f"Completed text extraction for {file_path}")
    return cleaned_text


def extract_text_from_images(page: PageObject) -> str:
    """
    Extracts text from a PDF page using Tesseract OCR.

    args:
        page (PageObject): PDF page object containing images  
    
    returns:
        str: Extracted text from images in the PDF page.
    """
    text = ""
    for image_obj in page.images:
        try:
            image = Image.open(io.BytesIO(image_obj.data))  
            ocr_text = pytesseract.image_to_string(image)
            text += ocr_text
            logger.info("extracted text from image")
        except Exception as e:
            logger.error(f"Error processing image: {e}") 
    return text
