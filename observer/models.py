
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from datetime import datetime


@dataclass
class SentimentResult:
    """
    Represents emotion detection output from sentiment analyzer.
    
    Attributes:
        label: Emotion label (e.g., 'joy', 'sadness', 'anger')
        score: Confidence score (0.0 to 1.0)
    """
    label: str
    score: float
    
    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "score": self.score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SentimentResult':
        return cls(
            label=data["label"],
            score=data["score"]
        )


@dataclass
class Preference:
    """
    Represents a user preference extracted from text.
    
    Attributes:
        topic: The subject of preference (e.g., 'morning', 'coffee', 'meetings')
        verb: Action indicating preference (e.g., 'like', 'hate', 'love')
        confidence: Confidence score (0.0 to 1.0), decays over time
        last_mentioned: Timestamp of last mention
    """
    topic: str
    verb: str
    confidence: float = 1.0
    last_mentioned: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "topic": self.topic,
            "verb": self.verb,
            "confidence": self.confidence,
            "last_mentioned": self.last_mentioned.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Preference':
        return cls(
            topic=data["topic"],
            verb=data["verb"],
            confidence=data.get("confidence", 1.0),
            last_mentioned=datetime.fromisoformat(data["last_mentioned"])
        )


@dataclass
class Fact:
    """
    Represents a memorable fact about the user.
    
    Fact Types:
        - relationship: Personal relationships (e.g., "friend John")
        - location: Location-based facts (e.g., "lives in Seattle")
        - life_event: Significant life events (e.g., "graduated in 2020")
    
    Attributes:
        fact_type: Type of fact ('relationship', 'location', 'life_event')
        data: Type-specific data dictionary
        confidence: Confidence score (0.0 to 1.0), decays over time
        last_mentioned: Timestamp of last mention
    """
    fact_type: Literal["relationship", "location", "life_event"]
    data: Dict[str, Optional[str]]  # Type-specific fields
    confidence: float = 1.0
    last_mentioned: datetime = field(default_factory=datetime.now)
    
    # Convenience properties for accessing type-specific data
    @property
    def person(self) -> Optional[str]:
        """For relationship facts"""
        return self.data.get("person")
    
    @property
    def relation(self) -> Optional[str]:
        """For relationship facts"""
        return self.data.get("relation")
    
    @property
    def place(self) -> Optional[str]:
        """For location facts"""
        return self.data.get("place")
    
    @property
    def context(self) -> Optional[str]:
        """For location facts"""
        return self.data.get("context")
    
    @property
    def event(self) -> Optional[str]:
        """For life_event facts"""
        return self.data.get("event")
    
    @property
    def date(self) -> Optional[str]:
        """For life_event facts"""
        return self.data.get("date")
    
    def to_dict(self) -> Dict:
        return {
            "fact_type": self.fact_type,
            "data": self.data,
            "confidence": self.confidence,
            "last_mentioned": self.last_mentioned.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Fact':
        return cls(
            fact_type=data["fact_type"],
            data=data["data"],
            confidence=data.get("confidence", 1.0),
            last_mentioned=datetime.fromisoformat(data["last_mentioned"])
        )
    
    @classmethod
    def from_raw_dict(cls, raw: Dict) -> 'Fact':
        fact_type = raw["type"]
        # Remove 'type' and use remaining fields as data
        data = {k: v for k, v in raw.items() if k != "type"}
        return cls(fact_type=fact_type, data=data)


@dataclass
class UserMemory:
    """
    Aggregates all user memory: sentiments, preferences, and facts.
    
    This structure is used by the memory store for persistence and 
    by the actor layer for context injection into prompts.
    
    Attributes:
        user_id: Unique identifier for the user
        recent_sentiments: List of recent emotion detections
        preferences: List of extracted preferences
        facts: List of extracted facts
        last_updated: Timestamp of last memory update
    """
    user_id: str = "default_user"
    recent_sentiments: List[SentimentResult] = field(default_factory=list)
    preferences: List[Preference] = field(default_factory=list)
    facts: List[Fact] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "recent_sentiments": [s.to_dict() for s in self.recent_sentiments],
            "preferences": [p.to_dict() for p in self.preferences],
            "facts": [f.to_dict() for f in self.facts],
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserMemory':
        return cls(
            user_id=data.get("user_id", "default_user"),
            recent_sentiments=[
                SentimentResult.from_dict(s) for s in data.get("recent_sentiments", [])
            ],
            preferences=[
                Preference.from_dict(p) for p in data.get("preferences", [])
            ],
            facts=[
                Fact.from_dict(f) for f in data.get("facts", [])
            ],
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
    
    def get_high_confidence_items(self, threshold: float = 0.5) -> 'UserMemory':
        """
        Returns a filtered UserMemory containing only high-confidence items.
        Useful for generating concise context for actor prompts.
        """
        return UserMemory(
            user_id=self.user_id,
            recent_sentiments=self.recent_sentiments,  # Always include recent emotions
            preferences=[p for p in self.preferences if p.confidence >= threshold],
            facts=[f for f in self.facts if f.confidence >= threshold],
            last_updated=self.last_updated
        )


__all__ = [
    'SentimentResult',
    'Preference', 
    'Fact',
    'UserMemory'
]
