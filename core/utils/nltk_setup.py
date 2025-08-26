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
    
    # Try to get punkt_tab (preferred) or punkt (fallback)
    punkt_resources = [
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('punkt', 'tokenizers/punkt')  # fallback
    ]
    
    punkt_downloaded = False
    for resource_name, check_path in punkt_resources:
        try:
            nltk.data.find(check_path)
            print(f"✅ NLTK resource '{resource_name}' already available")
            punkt_downloaded = True
            break
        except LookupError:
            try:
                print(f"📥 Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, download_dir=str(nltk_data_dir))
                print(f"✅ Successfully downloaded '{resource_name}'")
                punkt_downloaded = True
                break
            except Exception as e:
                print(f"❌ Failed to download '{resource_name}': {e}")
    
    if not punkt_downloaded:
        print("⚠️  No punkt tokenizer available - will use simple fallback")
    
    # Download stopwords
    try:
        nltk.data.find('corpora/stopwords')
        print("✅ NLTK resource 'stopwords' already available")
    except LookupError:
        try:
            print("📥 Downloading NLTK resource: stopwords")
            nltk.download('stopwords', download_dir=str(nltk_data_dir))
            print("✅ Successfully downloaded 'stopwords'")
        except Exception as e:
            print(f"❌ Failed to download 'stopwords': {e}")
    
    print(f"🎯 NLTK data directory: {nltk_data_dir}")
    
    # Test that punkt_tab is working
    try:
        from nltk.tokenize import word_tokenize
        test_tokens = word_tokenize("This is a test sentence.")
        print(f"✅ NLTK tokenizer test successful: {len(test_tokens)} tokens")
        return True
    except Exception as e:
        print(f"⚠️  NLTK tokenizer test failed: {e}")
        return False


if __name__ == "__main__":
    setup_nltk_data()