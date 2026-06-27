import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # Model settings
    MODEL_CONFIDENCE_THRESHOLD = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.5"))
    MODELS_AVAILABLE = ["DeepLabV3"]
    
    # App settings
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "5"))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() in ('true', '1', 't')
    
    # Logging
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_LEVEL = logging.DEBUG if DEBUG_MODE else logging.INFO

    @classmethod
    def check_keys(cls) -> bool:
        """
        Validates that all essential configuration keys have valid types/values.
        
        Returns:
            bool: True if configuration is valid, else False.
        """
        try:
            assert isinstance(cls.MODEL_CONFIDENCE_THRESHOLD, float)
            assert 0.0 <= cls.MODEL_CONFIDENCE_THRESHOLD <= 1.0
            assert isinstance(cls.MAX_IMAGE_SIZE_MB, int)
            assert cls.MAX_IMAGE_SIZE_MB > 0
            assert isinstance(cls.DEBUG_MODE, bool)
            return True
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False
