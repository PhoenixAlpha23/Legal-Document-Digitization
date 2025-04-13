import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import numpy as np
from PIL import Image
import logging
import sys
import fitz

from utils.config import Config
from models.ocr_processor import OCRProcessor
from models.yolo_detector import YOLODetector
from services.detection_service import load_detector, load_ocr_processor
from services.document_processing import process_image, process_pdf_page
from models.LLMchain import process_legal_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Legal Document Digitization with YOLO OCR",
    page_icon=":page_facing_up:",
    layout="wide"
)

def get_supported_languages():
    """Returns a dictionary of supported languages and their codes."""
    return {
        'English': 'eng',
        'Hindi': 'hin',
        'Marathi': 'mar'
    }

def main():
    detector = load_detector()
    ocr_processor = load_ocr_processor()
    
    st.title("Legal Document Digitizer")
    st.write("By Aryan Tandon and Umesh Tiwari")
    st.sidebar.title("Document Processing Options")

    st.sidebar.subheader("Language Settings")
    available_languages = get_supported_languages()
    default_lang = "English"

    primary_lang = st.sidebar.selectbox(
        "Primary Language",
        options=list(available_languages.keys()),
        index=list(available_languages.keys()).index(default_lang),
        help="Select the main language of your document",
    )
    
    additional_langs = st.sidebar.multiselect(
        "Additional Languages (Optional)",
        options=[lang for lang in available_languages.keys() if lang != primary_lang],
        help="Select additional languages if your document contains multiple languages",
    )

    selected_langs = [primary_lang] + additional_langs
    lang_codes = "+".join([available_languages[lang] for lang in selected_langs])

    psm = st.sidebar.selectbox(
        "Text Layout Detection",
        options=[3, 4, 6, 11, 12],
        index=0,
        format_func=lambda x: {
            3: "Automatic Detection",
            4: "Single Column Layout",
            6: "Single Text Block",
            11: "Line by Line",
            12: "Word by Word",
        }[x],
        help="Choose how the system should read your document's layout",
    )
    
    ocr_processor.update_config(lang_codes, psm)
#Preprocessing 
    st.sidebar.subheader("Image Enhancement Options")
    apply_threshold = st.sidebar.checkbox(
        "Sharpen Text", value=True, help="Improves text clarity by increasing contrast"
    )
    apply_deskew = st.sidebar.checkbox(
        "Straighten Document", value=True, help="Corrects tilted or skewed documents"
    )
    apply_denoise = st.sidebar.checkbox(
        "Remove Background Noise",
        value=True,
        help="Removes specks and background interference",
    )
    apply_contrast = st.sidebar.checkbox(
        "Enhance Text Visibility", value=False, help="Boosts text brightness and contrast"
    )

    preprocessing_options = {
        "apply_threshold": apply_threshold,
        "apply_deskew": apply_deskew,
        "apply_denoise": apply_denoise,
        "apply_contrast": apply_contrast,
    }

    uploaded_file = st.file_uploader(
        "Choose an image or PDF...", type=["jpg", "png", "jpeg", "pdf"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                process_pdf_document(uploaded_file, detector, ocr_processor, preprocessing_options)
            else: 
                process_image_document(uploaded_file, detector, ocr_processor, preprocessing_options)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(f"An error occurred: {e}")
            
def process_pdf_document(uploaded_file, detector, ocr_processor, preprocessing_options):
    """Handle PDF document processing and UI display"""
    # Progress bar
    progress_bar = st.progress(0)

    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        total_pages = doc.page_count

        for page_num in range(total_pages):
            progress_bar.progress((page_num + 1) / total_pages)

            # Process one page at a time
            page = doc[page_num]
            image_np, error = process_pdf_page(page, dpi=300)

            if error:
                st.error(f"Error processing page {page_num + 1}: {error}")
                continue

            if image_np is None:
                continue

            # Display the processed page
            st.image(image_np, caption=f"PDF Page {page_num+1}", width=400)

            # Detect objects in the page
            detections = detector.detect(image_np)
            image_with_boxes, text_images, table_images, stamp_images, signature_images = process_image(
                image_np,
                detections,
                ocr_processor,
                page_num,
                preprocessing_options,
            )

            display_processing_results(
                image_with_boxes, 
                detections, 
                text_images, 
                table_images, 
                stamp_images, 
                signature_images, 
                page_num + 1, 
                image_np, 
                ocr_processor, 
                preprocessing_options
            )

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        logger.exception(f"Error processing PDF: {e}")

    finally:
        # Clear the progress bar
        progress_bar.empty()

def process_image_document(uploaded_file, detector, ocr_processor, preprocessing_options):
    """Handle image document processing and UI display"""
    image = Image.open(uploaded_file).convert("RGB")
    image = np.array(image)
    st.image(image, caption="Uploaded Image", width=400)

    detections = detector.detect(image)
    image_with_boxes, text_images, table_images, stamp_images, signature_images = process_image(
        image, detections, ocr_processor, preprocessing_options=preprocessing_options
    )

    st.image(
        image_with_boxes, caption="Image with Detections and Labels", width=400
    )

    display_processing_results(
        image_with_boxes, 
        detections, 
        text_images, 
        table_images, 
        stamp_images, 
        signature_images, 
        None,  # No page number for single images
        image, 
        ocr_processor, 
        preprocessing_options
    )

def display_processing_results(image_with_boxes, detections, text_images, table_images, 
                             stamp_images, signature_images, page_num, 
                             original_image, ocr_processor, preprocessing_options):
    """Display processing results in the Streamlit UI"""
    
    page_info = f" (Page {page_num})" if page_num else ""
    
    st.subheader(f"Extracted Entities{page_info}")
    entity_counter = 1

    st.write(f"## Confidence Scores{page_info}:")
    with st.container():
        confidence_dict = {}
        for detection in detections:
            if "class" in detection:
                confidence_dict[detection["class"]] = detection["confidence"]

        st.write(f"1) Text: {confidence_dict.get('text', 'null')}")
        st.write(f"2) Table: {confidence_dict.get('table', 'null')}")
        st.write(f"3) Stamp: {confidence_dict.get('stamp', 'null')}")
        st.write(f"4) Signature: {confidence_dict.get('signature', 'null')}")

    if text_images:
        st.write("Text:")
        for img in text_images:
            st.write(f"{entity_counter})")
            st.image(img, width=400)
            entity_counter += 1
    else:
        st.write(f"{entity_counter}) Text: Not Detected")
        entity_counter += 1

    if table_images:
        st.write("Tables:")
        for img in table_images:
            st.write(f"{entity_counter})")
            st.image(img, width=400)
            entity_counter += 1
    else:
        st.write(f"{entity_counter}) Tables: Not Detected")
        entity_counter += 1

    if stamp_images:
        st.write("Stamps:")
        for img in stamp_images:
            st.write(f"{entity_counter})")
            st.image(img, width=400)
            entity_counter += 1
    else:
        st.write(f"{entity_counter}) Stamps: Not Detected")
        entity_counter += 1

    if signature_images:
        st.write("Signatures:")
        for img in signature_images:
            st.write(f"{entity_counter})")
            st.image(img, width=400)
            entity_counter += 1
    else:
        st.write(f"{entity_counter}) Signatures: Not Detected")
        entity_counter += 1

    try:
        # Extracted Text and LLM Analysis
        st.write("## Extracted Text:")
        combined_text = ""
        if text_images:
            for detection in detections:
                if "class" in detection and detection["class"] == "text":
                    ocr_results = ocr_processor.process_detections(original_image, [detection], preprocessing_options)
                    for result in ocr_results:
                        st.write(f"Text: {result['text']}")
                        combined_text += result['text'] + "\n"
            
            if combined_text.strip():
                st.subheader("LLM Analysis")
                with st.spinner("Analyzing text with LLM..."):
                    llm_results = process_legal_text(combined_text)
                    if "error" not in llm_results:
                        st.json(llm_results)
                    else:
                        st.error(llm_results["error"])
        else:
            st.write("No Text Detected")
    except Exception as e:
        st.error(f"An error occurred during text extraction or analysis: {e}")
        logger.exception(f"An error occurred during text extraction or analysis: {e}")

if __name__ == "__main__":
    main()
