import logging

logger = logging.getLogger(__name__)

def get_supported_languages():
    """Returns a dictionary of supported languages and their codes."""
    return {
        'English': 'eng',
        'Hindi': 'hin',
        'Marathi': 'mar'
    }

def get_psm_descriptions():
    """Returns a dictionary of PSM values and their descriptions."""
    return {
        3: "Automatic Detection",
        4: "Single Column Layout",
        6: "Single Text Block",
        11: "Line by Line",
        12: "Word by Word",
    }
