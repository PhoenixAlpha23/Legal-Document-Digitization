import streamlit as st
import logging
from models.yolo_detector import YOLODetector
from models.ocr_processor import OCRProcessor
from utils.config import Config

logger = logging.getLogger(__name__)

@st.cache_resource(max_entries=1)
def load_detector():
    """Load and cache the YOLO detector model."""
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
    """Load and cache the OCR processor."""
    with st.spinner("Loading Tesseract OCR engine..."):
        logger.info("Initializing Tesseract OCR processor")
        return OCRProcessor()
