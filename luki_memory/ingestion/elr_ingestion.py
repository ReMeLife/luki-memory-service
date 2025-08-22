"""
Enhanced ELR Ingestion Module

Advanced NLP processing for Electronic Life Records (ELR) with:
- Life story narrative parsing and extraction
- Structured data extraction from unstructured text
- Temporal event detection and timeline construction
- Relationship and entity extraction
- Sentiment and emotional tone analysis
- Memory trigger identification
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import spacy
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dateutil import parser as date_parser
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from ..schemas.elr import (
    ELRItem, ELRMetadata, ELRContentType, ConsentLevel, 
    SensitivityLevel, ELRProcessingResult
)
from .pipeline import ELRPipeline
from .chunker import ELRChunk

logger = logging.getLogger(__name__)


class LifeEventType(str, Enum):
    """Types of life events that can be extracted."""
    BIRTH = "birth"
    EDUCATION = "education"
    CAREER = "career"
    MARRIAGE = "marriage"
    FAMILY = "family"
    HEALTH = "health"
    TRAVEL = "travel"
    ACHIEVEMENT = "achievement"
    LOSS = "loss"
    RELATIONSHIP = "relationship"
    HOBBY = "hobby"
    MILESTONE = "milestone"


@dataclass
class LifeEvent:
    """Structured representation of a life event."""
    event_type: LifeEventType
    description: str
    date_mentioned: Optional[datetime] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    location: Optional[str] = None
    people_involved: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    source_text: str = ""


@dataclass
class PersonProfile:
    """Extracted person profile from ELR data."""
    name: Optional[str] = None
    relationship: Optional[str] = None
    characteristics: List[str] = field(default_factory=list)
    mentioned_activities: List[str] = field(default_factory=list)
    emotional_associations: List[str] = field(default_factory=list)
    frequency_mentioned: int = 0


@dataclass
class MemoryTrigger:
    """Identified memory trigger from ELR content."""
    trigger_type: str  # "music", "photo", "location", "person", "activity"
    trigger_value: str
    associated_memories: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0  # -1 to 1
    strength: float = 0.0  # 0 to 1


class EnhancedELRProcessor:
    """Enhanced ELR processor with advanced NLP capabilities."""
    
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        """Initialize enhanced ELR processor."""
        self.nlp = spacy.load(spacy_model)
        
        # Initialize sentiment analyzer
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.warning(f"Could not initialize NLTK sentiment analyzer: {e}")
            self.sentiment_analyzer = None
        
        # Initialize emotion classifier
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=-1  # CPU
            )
        except Exception as e:
            logger.warning(f"Could not initialize emotion classifier: {e}")
            self.emotion_classifier = None
        
        # Life event patterns
        self.life_event_patterns = self._compile_life_event_patterns()
        
        # Date extraction patterns
        self.date_patterns = self._compile_date_patterns()
        
        # Memory trigger keywords
        self.memory_triggers = self._load_memory_trigger_keywords()
    
    def _compile_life_event_patterns(self) -> Dict[LifeEventType, List[re.Pattern]]:
        """Compile regex patterns for life event detection."""
        patterns = {
            LifeEventType.BIRTH: [
                re.compile(r'\b(?:born|birth|came into the world)\b', re.IGNORECASE),
                re.compile(r'\b(?:was born|were born)\b', re.IGNORECASE),
            ],
            LifeEventType.EDUCATION: [
                re.compile(r'\b(?:school|university|college|graduated|studied|degree)\b', re.IGNORECASE),
                re.compile(r'\b(?:education|learning|academic)\b', re.IGNORECASE),
            ],
            LifeEventType.CAREER: [
                re.compile(r'\b(?:job|work|career|employed|profession|occupation)\b', re.IGNORECASE),
                re.compile(r'\b(?:started working|began career|retired)\b', re.IGNORECASE),
            ],
            LifeEventType.MARRIAGE: [
                re.compile(r'\b(?:married|wedding|spouse|husband|wife)\b', re.IGNORECASE),
                re.compile(r'\b(?:got married|tied the knot|wedding day)\b', re.IGNORECASE),
            ],
            LifeEventType.FAMILY: [
                re.compile(r'\b(?:children|kids|son|daughter|parent|family)\b', re.IGNORECASE),
                re.compile(r'\b(?:had a baby|became a parent|grandchild)\b', re.IGNORECASE),
            ],
            LifeEventType.HEALTH: [
                re.compile(r'\b(?:illness|disease|surgery|hospital|doctor|medical)\b', re.IGNORECASE),
                re.compile(r'\b(?:diagnosed|treatment|recovery|health)\b', re.IGNORECASE),
            ],
            LifeEventType.TRAVEL: [
                re.compile(r'\b(?:travel|trip|vacation|journey|visited|abroad)\b', re.IGNORECASE),
                re.compile(r'\b(?:went to|traveled to|holiday)\b', re.IGNORECASE),
            ],
            LifeEventType.ACHIEVEMENT: [
                re.compile(r'\b(?:award|achievement|success|accomplished|won)\b', re.IGNORECASE),
                re.compile(r'\b(?:proud|recognition|honor|medal)\b', re.IGNORECASE),
            ],
            LifeEventType.LOSS: [
                re.compile(r'\b(?:died|death|passed away|funeral|grief|loss)\b', re.IGNORECASE),
                re.compile(r'\b(?:lost|mourning|sad|departed)\b', re.IGNORECASE),
            ],
        }
        return patterns
    
    def _compile_date_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for date extraction."""
        return [
            re.compile(r'\b(?:in|during|around|about)\s+(\d{4})\b', re.IGNORECASE),
            re.compile(r'\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{2,4})\b'),
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE),
            re.compile(r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b', re.IGNORECASE),
            re.compile(r'\b(?:when I was|at age|aged)\s+(\d{1,2})\b', re.IGNORECASE),
        ]
    
    def _load_memory_trigger_keywords(self) -> Dict[str, List[str]]:
        """Load memory trigger keywords by category."""
        return {
            "music": ["song", "music", "singing", "piano", "guitar", "orchestra", "melody", "tune"],
            "photo": ["picture", "photograph", "photo", "image", "album", "snapshot"],
            "location": ["place", "home", "house", "garden", "park", "church", "school", "workplace"],
            "person": ["friend", "family", "mother", "father", "sister", "brother", "colleague"],
            "activity": ["hobby", "sport", "cooking", "reading", "walking", "dancing", "painting"],
            "sensory": ["smell", "taste", "sound", "touch", "sight", "aroma", "fragrance"],
        }
    
    def extract_life_events(self, text: str) -> List[LifeEvent]:
        """Extract structured life events from narrative text."""
        events = []
        doc = self.nlp(text)
        
        # Split text into sentences for better event isolation
        sentences = [sent.text for sent in doc.sents]
        
        for sentence in sentences:
            for event_type, patterns in self.life_event_patterns.items():
                for pattern in patterns:
                    if pattern.search(sentence):
                        event = self._create_life_event(
                            event_type, sentence, text
                        )
                        if event:
                            events.append(event)
                        break
        
        return self._deduplicate_events(events)
    
    def _create_life_event(self, event_type: LifeEventType, sentence: str, full_text: str) -> Optional[LifeEvent]:
        """Create a structured life event from a sentence."""
        try:
            # Extract dates
            dates = self._extract_dates_from_text(sentence)
            
            # Extract people
            people = self._extract_people_from_text(sentence)
            
            # Extract locations
            locations = self._extract_locations_from_text(sentence)
            
            # Analyze emotions
            emotions = self._analyze_emotions(sentence)
            
            # Calculate confidence based on specificity
            confidence = self._calculate_event_confidence(sentence, dates, people, locations)
            
            return LifeEvent(
                event_type=event_type,
                description=sentence.strip(),
                date_mentioned=dates[0] if dates else None,
                location=locations[0] if locations else None,
                people_involved=people,
                emotions=emotions,
                confidence_score=confidence,
                source_text=sentence
            )
            
        except Exception as e:
            logger.warning(f"Error creating life event: {e}")
            return None
    
    def _extract_dates_from_text(self, text: str) -> List[datetime]:
        """Extract dates from text using multiple approaches."""
        dates = []
        
        # Try regex patterns first
        for pattern in self.date_patterns:
            matches = pattern.findall(text)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        if len(match) == 1:  # Year only
                            date_str = match[0]
                        else:
                            date_str = '/'.join(match)
                    else:
                        date_str = match
                    
                    parsed_date = date_parser.parse(date_str, fuzzy=True)
                    dates.append(parsed_date)
                except:
                    continue
        
        # Try dateutil parser on the whole text
        try:
            parsed_date = date_parser.parse(text, fuzzy=True)
            if parsed_date not in dates:
                dates.append(parsed_date)
        except:
            pass
        
        return dates
    
    def _extract_people_from_text(self, text: str) -> List[str]:
        """Extract person names from text."""
        doc = self.nlp(text)
        people = []
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people.append(ent.text)
        
        # Also look for relationship terms
        relationship_terms = ["mother", "father", "sister", "brother", "friend", "colleague", "spouse", "husband", "wife"]
        for term in relationship_terms:
            if term.lower() in text.lower():
                people.append(term)
        
        return list(set(people))
    
    def _extract_locations_from_text(self, text: str) -> List[str]:
        """Extract location names from text."""
        doc = self.nlp(text)
        locations = []
        
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or location
                locations.append(ent.text)
        
        return locations
    
    def _analyze_emotions(self, text: str) -> List[str]:
        """Analyze emotions in text."""
        emotions = []
        
        # Use transformer-based emotion classifier if available
        if self.emotion_classifier:
            try:
                results = self.emotion_classifier(text)
                for result in results:
                    if result['score'] > 0.5:  # Confidence threshold
                        emotions.append(result['label'])
            except Exception as e:
                logger.warning(f"Error in emotion classification: {e}")
        
        # Use NLTK sentiment as fallback
        if self.sentiment_analyzer and not emotions:
            try:
                scores = self.sentiment_analyzer.polarity_scores(text)
                if scores['compound'] > 0.1:
                    emotions.append('positive')
                elif scores['compound'] < -0.1:
                    emotions.append('negative')
                else:
                    emotions.append('neutral')
            except Exception as e:
                logger.warning(f"Error in sentiment analysis: {e}")
        
        return emotions
    
    def _calculate_event_confidence(self, sentence: str, dates: List, people: List, locations: List) -> float:
        """Calculate confidence score for an extracted event."""
        confidence = 0.3  # Base confidence
        
        # Add confidence for specificity
        if dates:
            confidence += 0.3
        if people:
            confidence += 0.2
        if locations:
            confidence += 0.1
        
        # Add confidence for sentence length (more detail = higher confidence)
        word_count = len(sentence.split())
        if word_count > 10:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _deduplicate_events(self, events: List[LifeEvent]) -> List[LifeEvent]:
        """Remove duplicate events based on similarity."""
        if not events:
            return events
        
        unique_events = []
        for event in events:
            is_duplicate = False
            for existing in unique_events:
                # Simple similarity check based on description overlap
                if (event.event_type == existing.event_type and 
                    self._text_similarity(event.description, existing.description) > 0.8):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_events.append(event)
        
        return unique_events
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def extract_memory_triggers(self, text: str) -> List[MemoryTrigger]:
        """Extract potential memory triggers from text."""
        triggers = []
        doc = self.nlp(text)
        
        for trigger_type, keywords in self.memory_triggers.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Find sentences containing the trigger
                    for sent in doc.sents:
                        if keyword.lower() in sent.text.lower():
                            emotional_valence = self._get_emotional_valence(sent.text)
                            
                            trigger = MemoryTrigger(
                                trigger_type=trigger_type,
                                trigger_value=keyword,
                                associated_memories=[sent.text],
                                emotional_valence=emotional_valence,
                                strength=0.7  # Default strength
                            )
                            triggers.append(trigger)
        
        return triggers
    
    def _get_emotional_valence(self, text: str) -> float:
        """Get emotional valence (-1 to 1) of text."""
        if self.sentiment_analyzer:
            try:
                scores = self.sentiment_analyzer.polarity_scores(text)
                return scores['compound']
            except:
                pass
        return 0.0
    
    def create_person_profiles(self, text: str) -> List[PersonProfile]:
        """Extract and create profiles for people mentioned in the text."""
        doc = self.nlp(text)
        person_mentions = {}
        
        # Extract person entities
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                if name not in person_mentions:
                    person_mentions[name] = PersonProfile(name=name, frequency_mentioned=0)
                person_mentions[name].frequency_mentioned += 1
                
                # Extract context around the person mention
                sent = ent.sent
                person_mentions[name].mentioned_activities.extend(
                    self._extract_activities_from_sentence(sent.text)
                )
                person_mentions[name].emotional_associations.extend(
                    self._analyze_emotions(sent.text)
                )
        
        return list(person_mentions.values())
    
    def _extract_activities_from_sentence(self, sentence: str) -> List[str]:
        """Extract activities mentioned in a sentence."""
        doc = self.nlp(sentence)
        activities = []
        
        # Look for verbs that might indicate activities
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                activities.append(token.lemma_)
        
        return activities


class AdvancedELRIngestionPipeline(ELRPipeline):
    """Advanced ELR ingestion pipeline with enhanced NLP processing."""
    
    def __init__(self, spacy_model: str = "en_core_web_lg"):
        """Initialize advanced ELR ingestion pipeline."""
        super().__init__(spacy_model)
        self.enhanced_processor = EnhancedELRProcessor(spacy_model)
    
    def process_life_story_section(self, life_story_data: Dict, user_id: str, source_file: str = "") -> List[ELRItem]:
        """Process life story section with advanced NLP."""
        elr_items = []
        
        for section_key, narrative_text in life_story_data.items():
            if not isinstance(narrative_text, str) or not narrative_text.strip():
                continue
            
            # Extract life events
            life_events = self.enhanced_processor.extract_life_events(narrative_text)
            
            # Extract memory triggers
            memory_triggers = self.enhanced_processor.extract_memory_triggers(narrative_text)
            
            # Extract person profiles
            person_profiles = self.enhanced_processor.create_person_profiles(narrative_text)
            
            # Create main narrative ELR item
            main_metadata = ELRMetadata(
                user_id=user_id,
                source_file=source_file,
                content_type=ELRContentType.LIFE_STORY,
                consent_level=ConsentLevel.PRIVATE,
                sensitivity_level=SensitivityLevel.PERSONAL,
                chunk_index=None,
                total_chunks=None,
                language="en",
                tags=[section_key, "narrative", "life_story"],
                entities=[person.name for person in person_profiles if person.name],
                confidence_score=None,
                completeness_score=None,
                version=1,
                checksum=None
            )
            
            main_item = ELRItem(
                id=None,
                external_id=None,
                content=narrative_text,
                title=f"Life Story: {section_key.replace('_', ' ').title()}",
                metadata=main_metadata,
                parent_id=None,
                related_ids=[],
                event_date=None,
                date_range_start=None,
                date_range_end=None,
                location=None,
                coordinates=None,
                custom_fields={
                    "life_events": [event.__dict__ for event in life_events],
                    "memory_triggers": [trigger.__dict__ for trigger in memory_triggers],
                    "person_profiles": [profile.__dict__ for profile in person_profiles],
                    "section_type": section_key
                }
            )
            elr_items.append(main_item)
            
            # Create separate items for significant life events
            for event in life_events:
                if event.confidence_score > 0.7:  # High confidence events only
                    event_metadata = ELRMetadata(
                        user_id=user_id,
                        source_file=source_file,
                        content_type=ELRContentType.MEMORY,
                        consent_level=ConsentLevel.PRIVATE,
                        sensitivity_level=SensitivityLevel.PERSONAL,
                        chunk_index=None,
                        total_chunks=None,
                        language="en",
                        tags=["life_event", event.event_type.value] + event.emotions,
                        entities=event.people_involved,
                        confidence_score=event.confidence_score,
                        completeness_score=None,
                        version=1,
                        checksum=None
                    )
                    
                    event_item = ELRItem(
                        id=None,
                        external_id=None,
                        content=event.description,
                        title=f"{event.event_type.value.title()} Event",
                        metadata=event_metadata,
                        parent_id=main_item.id,
                        related_ids=[],
                        event_date=event.date_mentioned,
                        date_range_start=None,
                        date_range_end=None,
                        location=event.location,
                        coordinates=None,
                        custom_fields={
                            "event_type": event.event_type.value,
                            "people_involved": event.people_involved,
                            "emotions": event.emotions,
                            "extracted_from": section_key
                        }
                    )
                    elr_items.append(event_item)
        
        return elr_items
    
    def process_elr_data_enhanced(self, elr_data: Dict, user_id: str, source_file: str = "") -> ELRProcessingResult:
        """Enhanced ELR data processing with advanced NLP."""
        start_time = datetime.now()
        all_items = []
        errors = []
        
        try:
            # Process life story section with enhanced processing
            if "life_story" in elr_data:
                try:
                    life_story_items = self.process_life_story_section(
                        elr_data["life_story"], user_id, source_file
                    )
                    all_items.extend(life_story_items)
                except Exception as e:
                    error_msg = f"Error processing life story section: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            
            # Process other sections with standard processing
            other_sections = ["preferences", "memories", "family", "interests"]
            for section_name in other_sections:
                if section_name in elr_data:
                    try:
                        chunks = self._process_section(
                            section_name, elr_data[section_name], source_file
                        )
                        # Convert chunks to ELR items (simplified for now)
                        for chunk in chunks:
                            item_metadata = ELRMetadata(
                                user_id=user_id,
                                source_file=source_file,
                                content_type=ELRContentType.PREFERENCE if section_name == "preferences" else ELRContentType.MEMORY,
                                consent_level=ConsentLevel.PRIVATE,
                                sensitivity_level=SensitivityLevel.PERSONAL,
                                chunk_index=None,
                                total_chunks=None,
                                language="en",
                                tags=[section_name],
                                entities=[],
                                confidence_score=None,
                                completeness_score=None,
                                version=1,
                                checksum=None
                            )
                            
                            item = ELRItem(
                                id=None,
                                external_id=None,
                                content=chunk.text,
                                title=f"{section_name.title()}: {chunk.chunk_id}",
                                metadata=item_metadata,
                                parent_id=None,
                                related_ids=[],
                                event_date=None,
                                date_range_start=None,
                                date_range_end=None,
                                location=None,
                                coordinates=None,
                                custom_fields={}
                            )
                            all_items.append(item)
                            
                    except Exception as e:
                        error_msg = f"Error processing {section_name} section: {e}"
                        logger.error(error_msg)
                        errors.append(error_msg)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ELRProcessingResult(
                success=len(errors) == 0,
                processed_items=len(all_items),
                failed_items=len(errors),
                chunks_created=sum(1 for item in all_items if item.metadata.chunk_index is not None),
                errors=errors,
                processing_time_seconds=processing_time,
                created_item_ids=[item.id for item in all_items if item.id]
            )
            
        except Exception as e:
            error_msg = f"Failed to process ELR data: {e}"
            logger.error(error_msg)
            return ELRProcessingResult(
                success=False,
                processed_items=0,
                failed_items=1,
                errors=[error_msg],
                processing_time_seconds=(datetime.now() - start_time).total_seconds()
            )


def create_advanced_pipeline(spacy_model: str = "en_core_web_lg") -> AdvancedELRIngestionPipeline:
    """Factory function to create an advanced ELR ingestion pipeline."""
    return AdvancedELRIngestionPipeline(spacy_model)
