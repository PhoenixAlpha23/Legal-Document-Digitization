import cv2
import numpy as np
import streamlit as st
import pandas as pd
import fitz
import logging

logger = logging.getLogger(__name__)

def process_pdf_page(page, dpi=300):
    """
    Process a single PDF page and convert it to a numpy array.
    
    Args:
        page: fitz.Page object
        dpi: int, resolution for rendering (default: 300)
    
    Returns:
        tuple: (numpy array of the image, error message if any)
    """
    try:
        # Get the page's pixel matrix
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        
        # Convert to numpy array
        image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # If the image is CMYK (4 channels), convert to RGB (3 channels)
        if pix.n == 4:
            # Create RGB image
            image_rgb = np.zeros((pix.height, pix.width, 3), dtype=np.uint8)
            # Simple CMYK to RGB conversion
            image_rgb[:, :, 0] = image_np[:, :, 0] * (1 - image_np[:, :, 3] / 255.0)
            image_rgb[:, :, 1] = image_np[:, :, 1] * (1 - image_np[:, :, 3] / 255.0)
            image_rgb[:, :, 2] = image_np[:, :, 2] * (1 - image_np[:, :, 3] / 255.0)
            image_np = image_rgb

        return image_np, None
        
    except Exception as e:
        return None, str(e)

def process_image(image, detections, ocr_processor, page_num=None, preprocessing_options=None):
    """
    Process image with detections and extract entities.
    
    Args:
        image: numpy array of the image
        detections: list of detection objects
        ocr_processor: OCRProcessor instance
        page_num: page number (optional, for PDF)
        preprocessing_options: dict of preprocessing options
    
    Returns:
        tuple: (image with boxes, text images, table images, stamp images, signature images)
    """
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
                ocr_results = ocr_processor.process_detections(image, [detection], preprocessing_options)
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
