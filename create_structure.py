#!/usr/bin/env python3
"""
Create the EXACT directory structure from your documentation
Following your specified architecture
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create the exact directory structure from your documentation"""
    
    # Your exact structure from the documentation
    structure = {
        # Root files
        "": [
            "requirements.txt",
            "requirements-dev.txt", 
            "README.md",
            "streamlit_app.py",
            "config.py",
            ".env",
            ".gitignore"
        ],
        
        # Source code structure (exactly as you specified)
        "src": ["__init__.py"],
        "src/data": [
            "__init__.py",
            "data_loader.py",
            "market_data.py", 
            "preprocessor.py"
        ],
        "src/models": [
            "__init__.py",
            "llm_engine.py",
            "sentiment_analyzer.py",
            "portfolio_optimizer.py",
            "risk_manager.py"
        ],
        "src/analytics": [
            "__init__.py",
            "performance.py",
            "attribution.py",
            "reporting.py"
        ],
        "src/visualization": [
            "__init__.py",
            "charts.py",
            "dashboards.py",
            "power_bi_export.py"
        ],
        "src/utils": [
            "__init__.py",
            "logger.py",
            "helpers.py",
            "decorators.py"
        ],
        
        # Testing structure
        "tests": [
            "__init__.py",
            "test_data_loader.py",
            "test_models.py",
            "test_analytics.py",
            "test_visualization.py"
        ],
        
        # Data directories
        "data/raw": [".gitkeep"],
        "data/processed": [".gitkeep"],
        "data/models": [".gitkeep"],
        
        # Documentation
        "docs": [
            "API.md",
            "DEPLOYMENT.md", 
            "USER_GUIDE.md"
        ],
        
        # Streamlit configuration
        ".streamlit": ["config.toml"]
    }
    
    print("🚀 Creating directory structure...")
    
    for directory, files in structure.items():
        # Create directory
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}/")
        
        # Create files in directory
        for file in files:
            if directory:
                file_path = Path(directory) / file
            else:
                file_path = Path(file)
            
            # Don't overwrite existing files
            if not file_path.exists():
                file_path.touch()
                print(f"📄 Created file: {file_path}")
    
    print("\n🎉 Directory structure created successfully!")
    print("📁 Your project structure:")
    
    # Show the structure
    def show_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        path = Path(path)
        entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{entry.name}")
            
            if entry.is_dir() and not entry.name.startswith('.'):
                extension = "    " if is_last else "│   "
                show_tree(entry, prefix + extension, max_depth, current_depth + 1)
    
    show_tree(".", max_depth=3)

if __name__ == "__main__":
    create_directory_structure()