"""
AI Portfolio Management System - Configuration
Central configuration management for the entire application
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """
    Central configuration class for the AI Portfolio Management System
    Following enterprise-grade configuration management practices
    """
    
    # ============================================================================
    # Application Configuration
    # ============================================================================
    APP_NAME = "AI Portfolio Management System"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # ============================================================================
    # API Keys
    # ============================================================================
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # ============================================================================
    # Database Configuration
    # ============================================================================
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///portfolio.db")
    
    # ============================================================================
    # Model Configuration
    # ============================================================================
    DEFAULT_LLM_MODEL = "microsoft/DialoGPT-medium"
    FINBERT_MODEL = "ProsusAI/finbert"
    MAX_SEQUENCE_LENGTH = 512
    CACHE_EXPIRY_HOURS = 24
    
    # ============================================================================
    # Portfolio Configuration
    # ============================================================================
    DEFAULT_RISK_FREE_RATE = 0.02
    DEFAULT_MARKET_RETURN = 0.08
    REBALANCE_FREQUENCY = "monthly"
    MAX_POSITION_SIZE = 0.20
    MIN_POSITION_SIZE = 0.01
    
    # ============================================================================
    # Data Configuration
    # ============================================================================
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = DATA_DIR / "models"
    
    # Create directories if they don't exist
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # Streamlit Configuration
    # ============================================================================
    STREAMLIT_CONFIG = {
        "page_title": APP_NAME,
        "page_icon": "📊",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # ============================================================================
    # Azure Configuration (Power BI Integration)
    # ============================================================================
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_SERVICEBUS_CONNECTION_STRING = os.getenv("AZURE_SERVICEBUS_CONNECTION_STRING")
    
    # ============================================================================
    # Validation
    # ============================================================================
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """
        Validate configuration and return status
        """
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check required API keys
        required_keys = {
            "ALPHA_VANTAGE_API_KEY": cls.ALPHA_VANTAGE_API_KEY,
            "FRED_API_KEY": cls.FRED_API_KEY,
            "HUGGINGFACE_API_KEY": cls.HUGGINGFACE_API_KEY
        }
        
        for key_name, key_value in required_keys.items():
            if not key_value:
                validation_results["warnings"].append(f"Missing {key_name}")
        
        # Check optional keys
        optional_keys = {
            "OPENAI_API_KEY": cls.OPENAI_API_KEY,
            "AZURE_STORAGE_CONNECTION_STRING": cls.AZURE_STORAGE_CONNECTION_STRING
        }
        
        for key_name, key_value in optional_keys.items():
            if not key_value:
                validation_results["warnings"].append(f"Optional {key_name} not set")
        
        return validation_results
    
    # ============================================================================
    # Display Configuration
    # ============================================================================
    @classmethod
    def display_config(cls) -> None:
        """
        Display current configuration (for debugging)
        """
        print("=" * 60)
        print(f"🚀 {cls.APP_NAME} v{cls.APP_VERSION}")
        print("=" * 60)
        print(f"Debug Mode: {cls.DEBUG}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Database: {cls.DATABASE_URL}")
        print(f"Default LLM Model: {cls.DEFAULT_LLM_MODEL}")
        print(f"FinBERT Model: {cls.FINBERT_MODEL}")
        print("=" * 60)
        
        # Validation
        validation = cls.validate_config()
        if validation["warnings"]:
            print("⚠️  Warnings:")
            for warning in validation["warnings"]:
                print(f"   - {warning}")
        
        if validation["errors"]:
            print("❌ Errors:")
            for error in validation["errors"]:
                print(f"   - {error}")
        
        print("=" * 60)

# Create global config instance
config = Config()

# Validation on import
if __name__ == "__main__":
    config.display_config()
