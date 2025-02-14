import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import logging
import sys
import io
import cv2
import pandas as pd
import fitz

from utils.config import Config  # Make sure these paths are correct
from utils.pdf_processing import process_pdf  # Import the UPDATED process_pdf
from utils.image_processing import preprocess_image
from models.yolo_detector import YOLODetector

st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, language='eng', psm=3):
        self.tesseract_config = f'-l {language} --psm {psm}'
        import pytesseract
        self.pytesseract = pytesseract

    def process_detections(self, image, detections):
        results = []
        for detection in detections:
            bbox = detection['bbox']
            roi = self.extract_roi(image, bbox)

            try:
                text = self.pytesseract.image_to_string(roi, config=self.tesseract_config)
            except Exception as e:
                logger.error(f"Tesseract processing error: {e}")
                text = ''

            results.append({
                'bbox': bbox,
                'text': text,
                'corrected_text': text
            })
        return results

    @staticmethod
    def extract_roi(image, bbox):
        x, y, w, h = bbox
        return image[int(y):int(y + h), int(x):int(x + w)]

@st.cache_resource(max_entries=1)
def load_detector():
    with st.spinner("Loading YOLO model..."):
        logger.info("Initializing YOLO model...")
        try:
            detector = YOLODetector(Config.model_path)
            logger.info("YOLO model initialized successfully.")
            return detector
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            st.error(f"Error loading YOLO model: {e}")
            raise

@st.cache_resource(max_entries=1)
def load_ocr_processor():
    with st.spinner("Loading Tesseract OCR engine..."):
        logger.info("Initializing Tesseract OCR processor")
        return OCRProcessor()

def process_image(image, detections, ocr_processor, page_num=None):
    image_with_boxes = image.copy()
    text_images = []
    table_images = []
    stamp_images = []
    signature_images = []

    if detections:
        for detection in detections:
            bbox = detection['bbox']
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if 'class' in detection:
                category = detection['class']
            elif 'confidence' in detection:
                confidence = detection['confidence']
                if confidence > 0.8:
                    category = "text"
                else:
                    category = "unknown"
            else:
                category = "unknown"

            cv2.putText(image_with_boxes, str(category), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            roi = image[y:y+h, x:x+w]

            if category == "text":
                ocr_results = ocr_processor.process_detections(image, [detection])
                for result in ocr_results:
                    st.write(f"Category: {category}, Text: {result['text']}")
                text_images.append(roi)
            elif category == "table":
                try:
                    df = pd.DataFrame()  # Placeholder - Replace with actual conversion
                    st.dataframe(df)
                    table_images.append(roi)
                except Exception as e:
                    st.error(f"Error processing table: {e}")
                    logger.exception(f"Error processing table: {e}")
            elif category == "stamp":
                st.write("Stamp detected!")
                stamp_images.append(roi)
            elif category == "signature":
                st.write("Signature detected!")
                signature_images.append(roi)
            else:
                st.write(f"Category: {category} (Unknown)")
    return image_with_boxes, text_images, table_images, stamp_images, signature_images

def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()
    st.title("Legal Document digitizer")
    st.write("By Aryan Tandon and Umesh Tiwari")

    uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "png", "jpeg", "pdf"])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                options = {  # Define your image preprocessing options here!
                    "resize": (800, 1000),  # Example: Resize the image (adjust as needed)
                    "grayscale": False,      # Example: Convert to grayscale (adjust as needed)
                    # ... add other preprocessing options as needed
                }

                extracted_text_per_page = process_pdf(uploaded_file, options)

                if extracted_text_per_page:
                    for page_num, text in enumerate(extracted_text_per_page):
                        st.subheader(f"Page {page_num + 1} Extracted Text:")
                        st.write(text)  # Or use st.text_area(text, height=200)

                        doc = fitz.open(uploaded_file)
                        page = doc[page_num]
                        pix = page.get_pixmap()
                        image = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
                        st.image(image, caption=f"PDF Page {page_num + 1}")
                        doc.close()

                else:
                    st.write("No text extracted from the PDF.")

            else:  # It's an image (this part remains largely the same)
                image = Image.open(uploaded_file).convert("RGB")
                image = np.array(image)
                st.image(image, caption="Uploaded Image")

                detections = detector.detect(image)
                image_with_boxes, text_images, table_images, stamp_images, signature_images = process_image(image, detections, ocr_processor)
                st.image(image_with_boxes, caption="Image with Detections and Labels")

                st.subheader("Extracted Entities")
                entity_counter = 1

                st.write("## Confidence Scores:")
                with st.container():
                    confidence_dict = {}
                    for detection in detections:
                        if 'class' in detection:
                            confidence_dict[detection['class']] = detection['confidence']

                    st.write(f"1) Text: {confidence_dict.get('text', 'null')}")
                    st.write(f"2) Table: {confidence_dict.get('table', 'null')}")
                    st.write(f"3) Stamp: {confidence_dict.get('stamp', 'null')}")
                    st.write(f"4) Signature: {confidence_dict.get('signature', 'null')}")

                if text_images:
                    st.write("Text:")
                    for img in text_images:
                        st.write(f"{entity_counter})")
                        st.image(img)
                        entity_counter += 1
                else:
                    st.write(f"{entity_counter}) Text: Not Detected")
                    entity_counter += 1

                if table_images:
                    st.write("Tables:")
                    for img in table_images:
                        st.write(f"{entity_counter})")
                        st.image(img)
                        entity_counter += 1
                else:
                    st.write(f"{entity_counter}) Tables: Not Detected")
                    entity_counter += 1

                if stamp_images:
                    st.write("Stamps:")
                    for img in stamp_images:
                        st.write(f"{entity_counter})")
                        st.image(img)
                        entity_counter += 1
                else:
                    st.write(f"{entity_counter}) Stamps: Not Detected")
                    entity_counter += 1

                if signature_images:
                    st.write("Signatures:")
                    for img in signature_images:
                        st.write(f"{entity_counter})")
                        st.image(img)
                        entity_counter += 1
                else:
                    st.write(f"{entity_counter}) Signatures: Not Detected")
                    entity_counter += 1

                st.write("## Extracted Text:")

                if text_images:
                    for detection in detections:
                        if 'class' in detection and detection['class'] == 'text':
                            ocr_results = ocr_processor.process_detections(image, [detection])
                            for result in ocr_results:
                                st.write(f"Text: {result['text']}")
                else:
                    st.write("No Text Detected")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
