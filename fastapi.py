from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from PIL import Image
import numpy as np
import fitz
import io
import logging
import sys

from utils.config import Config
from models.ocr_processor import OCRProcessor
from models.yolo_detector import YOLODetector
from services.detection_service import load_detector, load_ocr_processor
from services.document_processing import process_image, process_pdf_page
from models.LLMchain import process_legal_text

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = load_detector()
ocr_processor = load_ocr_processor()

SUPPORTED_LANGUAGES = {
    'English': 'eng',
    'Hindi': 'hin',
    'Marathi': 'mar'
}

@app.post("/process-document")
async def process_document(
    file: UploadFile = File(...),
    primary_lang: str = Form("English"),
    additional_langs: Optional[List[str]] = Form([]),
    psm: int = Form(3),
    apply_threshold: bool = Form(True),
    apply_deskew: bool = Form(True),
    apply_denoise: bool = Form(True),
    apply_contrast: bool = Form(False)
):
    try:
        selected_langs = [primary_lang] + (additional_langs or [])
        lang_codes = "+".join([SUPPORTED_LANGUAGES[lang] for lang in selected_langs if lang in SUPPORTED_LANGUAGES])
        ocr_processor.update_config(lang_codes, psm)

        preprocessing_options = {
            "apply_threshold": apply_threshold,
            "apply_deskew": apply_deskew,
            "apply_denoise": apply_denoise,
            "apply_contrast": apply_contrast
        }

        content = await file.read()

        if file.content_type == "application/pdf":
            return await process_pdf_bytes(content, preprocessing_options)
        else:
            image = Image.open(io.BytesIO(content)).convert("RGB")
            image_np = np.array(image)
            return await process_image_bytes(image_np, preprocessing_options)

    except Exception as e:
        logger.exception("Processing error")
        return JSONResponse(status_code=500, content={"error": str(e)})

def extract_text_and_analyze(detections, image_np, preprocessing_options):
    text_images = [d for d in detections if d["class"] == "text"]
    combined_text = ""
    for d in text_images:
        results = ocr_processor.process_detections(image_np, [d], preprocessing_options)
        for r in results:
            combined_text += r['text'] + "\n"

    if not combined_text.strip():
        return {"message": "No text for analysis"}

    return process_legal_text(combined_text)

async def process_pdf_bytes(content: bytes, preprocessing_options):
    doc = fitz.open(stream=content, filetype="pdf")
    page = doc[0]
    image_np, error = process_pdf_page(page, dpi=300)
    if error or image_np is None:
        return JSONResponse(status_code=400, content={"error": error or "Unable to read PDF"})

    detections = detector.detect(image_np)
    _, text_images, table_images, stamp_images, signature_images = process_image(
        image_np, detections, ocr_processor, 0, preprocessing_options
    )

    llm_output = extract_text_and_analyze(detections, image_np, preprocessing_options)

    response = {
        "detected": {
            "text": len(text_images) > 0,
            "table": len(table_images) > 0,
            "stamp": len(stamp_images) > 0,
            "signature": len(signature_images) > 0,
        },
        "llm_analysis": llm_output,
    }
    doc.close()
    return response

async def process_image_bytes(image_np, preprocessing_options):
    detections = detector.detect(image_np)
    _, text_images, table_images, stamp_images, signature_images = process_image(
        image_np, detections, ocr_processor, preprocessing_options=preprocessing_options
    )

    llm_output = extract_text_and_analyze(detections, image_np, preprocessing_options)

    response = {
        "detected": {
            "text": len(text_images) > 0,
            "table": len(table_images) > 0,
            "stamp": len(stamp_images) > 0,
            "signature": len(signature_images) > 0,
        },
        "llm_analysis": llm_output,
    }
    return response
