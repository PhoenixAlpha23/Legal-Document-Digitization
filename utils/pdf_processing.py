from PIL import Image
import numpy as np
from .image_processing import preprocess_image
from .ocr_processor import OCRProcessor
import fitz  # For PDF processing

# Initialize OCR Processor (outside the function for efficiency)
ocr_processor = OCRProcessor(language="eng+hin+mar", psm=6)  # Or whatever languages you need

def process_pdf(uploaded_file, options):
    """
    Process an uploaded PDF file:
    1. Convert PDF to images (using fitz)
    2. Preprocess images
    3. Extract text using OCR
    """
    try:
        doc = fitz.open(uploaded_file)
        extracted_text = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            pix = page.get_pixmap() # Get image of the page
            img = Image.open(io.BytesIO(pix.tobytes())).convert("RGB") # Open with PIL

            preprocessed_img = preprocess_image(img, options)  # Assuming options are defined
            text = ocr_processor.extract_text(preprocessed_img) # Your OCR function
            extracted_text.append(text)
        doc.close() # Close the document
        return extracted_text

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []
