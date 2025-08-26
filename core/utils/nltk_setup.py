"""
NLTK setup utility to ensure required resources are downloaded.
Run this once to set up NLTK data for the application.
"""
import nltk
import os
from pathlib import Path


def setup_nltk_data():
    """Download required NLTK data if not already present."""
    
    # Set NLTK data path to a local directory within the project
    nltk_data_dir = Path(__file__).parent.parent.parent / "data" / "nltk_data"
    nltk_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Add our custom path to NLTK's data path
    nltk.data.path.insert(0, str(nltk_data_dir))
    
    required_resources = [
        'punkt_tab',
        'punkt',
        'stopwords'
    ]
    
    for resource in required_resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"✅ NLTK resource '{resource}' already available")
        except LookupError:
            try:
                print(f"📥 Downloading NLTK resource: {resource}")
                nltk.download(resource, download_dir=str(nltk_data_dir))
                print(f"✅ Successfully downloaded '{resource}'")
            except Exception as e:
                print(f"❌ Failed to download '{resource}': {e}")
    
    print(f"🎯 NLTK data directory: {nltk_data_dir}")


if __name__ == "__main__":
    setup_nltk_data()