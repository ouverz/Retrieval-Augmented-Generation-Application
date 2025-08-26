#!/usr/bin/env python3
"""
Setup script to download required NLTK data.
Run this after installing requirements.txt
"""
import sys
from pathlib import Path

# Add core to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils.nltk_setup import setup_nltk_data

if __name__ == "__main__":
    print("🔧 Setting up NLTK data for BM25 tokenization...")
    setup_nltk_data()
    print("✅ NLTK setup complete!")