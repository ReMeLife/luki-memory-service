"""
PII/Sensitive-Field Removal Module
Extract entities and redact sensitive information from text.

Responsibilities:
- Extract named entities using spaCy
- Identify and redact PII (Personal Identifiable Information)
- Handle sensitive field removal
- Analyze sentiment (placeholder for sentiment analysis)
"""

import logging
from typing import Dict, List, Optional

import spacy
from spacy.lang.en import English

logger = logging.getLogger(__name__)


class RedactionError(Exception):
    """Custom exception for redaction errors."""
    pass


class TextRedactor:
    """Handles PII detection and redaction."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize text redactor.
        
        Args:
            spacy_model: Name of spaCy model to use for NLP processing
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model {spacy_model} not found, using basic English")
            self.nlp = English()
        
        # Add sentencizer component if not present
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text (placeholder for sentiment analysis).
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sentiment scores dictionary
        """
        # TODO: Implement proper sentiment analysis
        # For MVP, return neutral sentiment
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "confidence": 0.5
        }
    
    def redact_pii(self, text: str, entity_types: Optional[List[str]] = None) -> str:
        """
        Redact PII from text based on entity types.
        
        Args:
            text: Input text to redact
            entity_types: List of entity types to redact (default: common PII types)
            
        Returns:
            Text with PII redacted
        """
        if entity_types is None:
            entity_types = ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "PERCENT"]
        
        doc = self.nlp(text)
        redacted_text = text
        
        # Sort entities by position (reverse order to maintain indices)
        entities = sorted(doc.ents, key=lambda x: x.start_char, reverse=True)
        
        for ent in entities:
            if ent.label_ in entity_types:
                redacted_text = (
                    redacted_text[:ent.start_char] + 
                    f"[{ent.label_}]" + 
                    redacted_text[ent.end_char:]
                )
        
        return redacted_text
    
    def redact_sensitive_terms(self, text: str) -> str:
        """
        Apply additional redaction for highly sensitive content.
        
        Args:
            text: Input text to redact
            
        Returns:
            Text with sensitive terms redacted
        """
        # For confidential content, redact additional entity types
        sensitive_entity_types = [
            "PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "PERCENT",
            "CARDINAL", "ORDINAL", "QUANTITY", "PHONE", "EMAIL"
        ]
        
        return self.redact_pii(text, sensitive_entity_types)


def create_redactor(spacy_model: str = "en_core_web_sm") -> TextRedactor:
    """
    Factory function to create a text redactor.
    
    Args:
        spacy_model: spaCy model to use
        
    Returns:
        Initialized TextRedactor instance
    """
    return TextRedactor(spacy_model)
