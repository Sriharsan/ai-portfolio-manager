# config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    
    # Print loaded environment variables for debugging
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Debug API key loading
    for key in ['ALPHA_VANTAGE_API_KEY', 'FRED_API_KEY', 'OPENAI_API_KEY', 'HUGGINGFACE_API_KEY']:
        value = os.getenv(key, '')
        logger.info(f"{key}: {'SET' if value else 'NOT SET'} (length: {len(value)})")
    
    # App basics
    APP_NAME = "AI Portfolio Manager"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # API Keys (with fallbacks)
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Portfolio defaults
    DEFAULT_RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.02"))
    MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.2"))
    MIN_POSITION_SIZE = float(os.getenv("MIN_POSITION_SIZE", "0.01"))
    
    # Cache settings
    CACHE_DURATION_HOURS = int(os.getenv("CACHE_DURATION_HOURS", "1"))
    
    # Rate limiting
    API_CALLS_PER_MINUTE = int(os.getenv("API_CALLS_PER_MINUTE", "60"))
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    
    # Directories
    DATA_DIR = Path("data")
    CACHE_DIR = DATA_DIR / "cache"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        warnings = []
        
        if not cls.ALPHA_VANTAGE_API_KEY:
            warnings.append("Missing ALPHA_VANTAGE_API_KEY - some features may be limited")
        
        if not cls.FRED_API_KEY:
            warnings.append("Missing FRED_API_KEY - economic data unavailable")
        
        if not cls.HUGGINGFACE_API_KEY:
            warnings.append("Missing HUGGINGFACE_API_KEY - using fallback AI models")
        
        return {"warnings": warnings, "status": "ok"}
    
    @classmethod 
    def get_api_status(cls):
        status = {
            "alpha_vantage": bool(cls.ALPHA_VANTAGE_API_KEY),
            "fred": bool(cls.FRED_API_KEY),
            "huggingface": bool(cls.HUGGINGFACE_API_KEY),
            "openai": bool(cls.OPENAI_API_KEY)
        }
    
        # Log status for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"API Status: {status}")
    
        return status

# Global config instance
config = Config()

# Validate on import
validation = config.validate_config()
if validation["warnings"]:
    import warnings
    for warning in validation["warnings"]:
        warnings.warn(warning, UserWarning)