#!/usr/bin/env python3
"""
Docker-safe NLTK data download script.
Downloads required NLTK data to a specified directory.
"""
import nltk
import os
import sys
from pathlib import Path

def download_nltk_data():
    """Download NLTK data for production use."""
    
    # Set download directory
    download_dir = os.getenv('NLTK_DATA', '/app/data/nltk_data')
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.insert(0, download_dir)
    
    # Required resources
    resources = ['punkt', 'punkt_tab', 'stopwords']
    
    success_count = 0
    for resource in resources:
        try:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=download_dir, quiet=False)
            success_count += 1
            print(f"✅ Successfully downloaded {resource}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to download {resource}: {e}")
    
    print(f"📦 Downloaded {success_count}/{len(resources)} NLTK resources to {download_dir}")
    
    # Test the downloads
    try:
        from nltk.tokenize import word_tokenize
        test_tokens = word_tokenize("This is a test sentence.")
        print(f"✅ NLTK tokenization test passed: {test_tokens}")
        return True
    except Exception as e:
        print(f"❌ NLTK tokenization test failed: {e}")
        return False

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)