"""
AI Portfolio Management System - Optimized Configuration
Minimal memory footprint configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # App basics
    APP_NAME = "AI Portfolio Manager"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # Portfolio defaults
    DEFAULT_RISK_FREE_RATE = 0.02
    MAX_POSITION_SIZE = 0.2
    MIN_POSITION_SIZE = 0.01
    
    # Directories
    DATA_DIR = Path("data")
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        warnings = []
        
        if not cls.ALPHA_VANTAGE_API_KEY:
            warnings.append("Missing ALPHA_VANTAGE_API_KEY")
        if not cls.FRED_API_KEY:
            warnings.append("Missing FRED_API_KEY")
        if not cls.HUGGINGFACE_API_KEY:
            warnings.append("Missing HUGGINGFACE_API_KEY")
        
        return {"warnings": warnings}

config = Config()