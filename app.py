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

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Configure the Streamlit page
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

    # Sidebar for options
    st.sidebar.title("Document Processing Options")

    # Language Settings
    st.sidebar.subheader("Language Settings")
    available_languages = get_supported_languages()
    default_lang = "English"

    # Primary language selection
    primary_lang = st.sidebar.selectbox(
        "Primary Language",
        options=list(available_languages.keys()),
        index=list(available_languages.keys()).index(default_lang),
        help="Select the main language of your document",
    )

    # Additional languages selection
    additional_langs = st.sidebar.multiselect(
        "Additional Languages (Optional)",
        options=[lang for lang in available_languages.keys() if lang != primary_lang],
        help="Select additional languages if your document contains multiple languages",
    )

    # Combine selected languages for Tesseract
    selected_langs = [primary_lang] + additional_langs
    lang_codes = "+".join([available_languages[lang] for lang in selected_langs])

    # PSM Selection
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

    # Update OCR processor with selected language and PSM
    ocr_processor.update_config(lang_codes, psm)

    # Preprocessing options with better labels
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
            else:  # It's an image
                process_image_document(uploaded_file, detector, ocr_processor, preprocessing_options)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.exception(f"An error occurred: {e}")
            
def process_pdf_document(uploaded_file, detector, ocr_processor, preprocessing_options):
    """Handle PDF document processing and UI display for first page only"""
    try:
        # Use the improved PDF processing
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        
        # Process only the first page
        page = doc[0]
        image_np, error = process_pdf_page(page, dpi=300)

        if error:
            st.error(f"Error processing PDF page: {error}")
            return

        if image_np is None:
            st.error("Could not extract page from PDF")
            return

        # Display the processed page
        st.image(image_np, caption="PDF First Page", width=400)

        # Detect objects in the page
        detections = detector.detect(image_np)
        image_with_boxes, text_images, table_images, stamp_images, signature_images = process_image(
            image_np,
            detections,
            ocr_processor,
            0,  # First page
            preprocessing_options,
        )

        # Display the image with boxes
        st.image(image_with_boxes, caption="Image with Detections and Labels", width=400)

        # Display simplified results
        display_simplified_results(
            detections, 
            text_images, 
            table_images, 
            stamp_images, 
            signature_images, 
            image_np, 
            ocr_processor, 
            preprocessing_options
        )

        # Close the document
        doc.close()

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        logger.exception(f"Error processing PDF: {e}")

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

    display_simplified_results(
        detections, 
        text_images, 
        table_images, 
        stamp_images, 
        signature_images, 
        image, 
        ocr_processor, 
        preprocessing_options
    )

def display_simplified_results(detections, text_images, table_images, 
                             stamp_images, signature_images, 
                             original_image, ocr_processor, preprocessing_options):
    """Display simplified processing results in the Streamlit UI"""
    
    # Display confidence scores
    st.subheader("Confidence Scores:")
    with st.container():
        confidence_dict = {}
        for detection in detections:
            if "class" in detection:
                confidence_dict[detection["class"]] = detection["confidence"]

        cols = st.columns(4)
        with cols[0]:
            st.write(f"Text: {confidence_dict.get('text', 'null')}")
        with cols[1]:
            st.write(f"Table: {confidence_dict.get('table', 'null')}")
        with cols[2]:
            st.write(f"Stamp: {confidence_dict.get('stamp', 'null')}")
        with cols[3]:
            st.write(f"Signature: {confidence_dict.get('signature', 'null')}")

    # Display detected entities summary
    st.subheader("Entities Detected:")
    
    entity_summary = {
        "Text": len(text_images) > 0,
        "Tables": len(table_images) > 0,
        "Stamps": len(stamp_images) > 0,
        "Signatures": len(signature_images) > 0
    }
    
    cols = st.columns(4)
    for i, (entity_type, detected) in enumerate(entity_summary.items()):
        with cols[i]:
            status = "✅ Detected" if detected else "❌ Not Found"
            st.write(f"{entity_type}: {status}")
    
    # Process text for LLM analysis without showing OCR results
    try:
        # Only perform LLM analysis if text is detected
        if text_images:
            combined_text = ""
            for detection in detections:
                if "class" in detection and detection["class"] == "text":
                    ocr_results = ocr_processor.process_detections(original_image, [detection], preprocessing_options)
                    for result in ocr_results:
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
                st.info("No text content available for LLM analysis")
        else:
            st.info("No text detected for LLM analysis")
    except Exception as e:
        st.error(f"An error occurred during LLM analysis: {e}")
        logger.exception(f"An error occurred during LLM analysis: {e}")

if __name__ == "__main__":
    main()
