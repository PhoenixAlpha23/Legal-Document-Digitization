import logging
from utils.image_processing import preprocess_image

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, language='eng', psm=3):
        self.tesseract_config = f'-l {language} --psm {psm}'
        import pytesseract
        self.pytesseract = pytesseract
        
    def update_config(self, language, psm):
        """Update Tesseract configuration with new language and PSM."""
        self.tesseract_config = f'-l {language} --psm {psm}'

    def process_detections(self, image, detections, preprocessing_options=None):
        results = []
        for detection in detections:
            bbox = detection['bbox']
            roi = self.extract_roi(image, bbox)

            # Preprocess the ROI
            preprocessed_roi = preprocess_image(roi, preprocessing_options)

            try:
                text = self.pytesseract.image_to_string(preprocessed_roi, config=self.tesseract_config)
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
