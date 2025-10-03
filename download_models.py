#!/usr/bin/env python3
"""
Download required models and data for the memory service.
This script should be run during deployment to ensure all models are available.
"""

import logging
import sys
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_spacy_model():
    """Download the spaCy English model."""
    try:
        import spacy
        # Try to load the model first
        try:
            spacy.load("en_core_web_sm")
            logger.info("spaCy model en_core_web_sm already available")
            return True
        except OSError:
            logger.info("Downloading spaCy model en_core_web_sm...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            logger.info("spaCy model downloaded successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to download spaCy model: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk
        logger.info("Downloading NLTK data...")
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        return False

def main():
    """Main function to download all required models."""
    logger.info("Starting model downloads...")
    
    success = True
    
    # Download spaCy model
    if not download_spacy_model():
        success = False
    
    # Download NLTK data
    if not download_nltk_data():
        success = False
    
    if success:
        logger.info("All models downloaded successfully")
        sys.exit(0)
    else:
        logger.error("Some models failed to download")
        sys.exit(1)

if __name__ == "__main__":
    main()
