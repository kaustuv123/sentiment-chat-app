"""
Observer Layer Package

Provides sentiment analysis, preference extraction, and fact extraction 
capabilities along with standardized data models.
"""

# Import data models from models.py
from .models import (
    SentimentResult,
    Preference,
    Fact,
    UserMemory
)

# Re-export for convenient imports: from observer import SentimentResult
__all__ = [
    'SentimentResult',
    'Preference',
    'Fact',
    'UserMemory'
]
