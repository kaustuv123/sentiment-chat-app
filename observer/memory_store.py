"""
Memory Store for Sentiment Chat App

Handles persistence and retrieval of user memory (sentiments, preferences, facts)
with confidence decay logic to prioritize recent interactions.
"""

import os
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from .models import UserMemory, SentimentResult, Preference, Fact


class MemoryStore:
    """
    Manages user memory with JSON persistence and confidence decay.
    
    Features:
    - Persists UserMemory to data/memory/{user_id}.json
    - Applies confidence decay to existing items over time
    - Merges new extractions with existing memory
    - Provides formatted context strings for LLM prompts
    
    Attributes:
        DECAY_RATE: Confidence multiplier applied to existing items (0.95)
        MAX_RECENT_SENTIMENTS: Maximum number of sentiment results to keep (10)
        memory_dir: Directory for storing memory JSON files
    """
    
    DECAY_RATE = 0.95
    MAX_RECENT_SENTIMENTS = 10
    
    def __init__(self, memory_dir: str = "data/memory"):
        """
        Initialize memory store.
        
        Args:
            memory_dir: Directory path for storing user memory files
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_memory_path(self, user_id: str) -> Path:
        """Get file path for user's memory."""
        return self.memory_dir / f"{user_id}.json"
    
    def load(self, user_id: str) -> UserMemory:
        """
        Load user memory from disk.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            UserMemory object (empty if file doesn't exist)
        """
        memory_path = self._get_memory_path(user_id)
        
        if not memory_path.exists():
            return UserMemory(user_id=user_id)
        
        try:
            with open(memory_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return UserMemory.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Error loading memory for {user_id}: {e}")
            return UserMemory(user_id=user_id)
    
    def save(self, memory: UserMemory) -> None:
        """
        Save user memory to disk.
        
        Args:
            memory: UserMemory object to persist
        """
        memory_path = self._get_memory_path(memory.user_id)
        
        try:
            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump(memory.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error saving memory for {memory.user_id}: {e}")
    
    def _apply_decay(self, memory: UserMemory) -> None:
        """
        Apply confidence decay to all preferences and facts.
        
        Multiplies confidence by DECAY_RATE for each item.
        Items with confidence < 0.1 are removed.
        
        Args:
            memory: UserMemory to apply decay to (modified in-place)
        """
        # Decay preferences
        memory.preferences = [
            Preference(
                topic=p.topic,
                verb=p.verb,
                confidence=p.confidence * self.DECAY_RATE,
                last_mentioned=p.last_mentioned
            )
            for p in memory.preferences
            if p.confidence * self.DECAY_RATE >= 0.1  # Remove low-confidence items
        ]
        
        # Decay facts
        memory.facts = [
            Fact(
                fact_type=f.fact_type,
                data=f.data,
                confidence=f.confidence * self.DECAY_RATE,
                last_mentioned=f.last_mentioned
            )
            for f in memory.facts
            if f.confidence * self.DECAY_RATE >= 0.1  # Remove low-confidence items
        ]
    
    def _merge_preferences(
        self, 
        existing: List[Preference], 
        new: List[Preference]
    ) -> List[Preference]:
        """
        Merge new preferences with existing ones.
        
        Logic:
        - If topic+verb already exists: update confidence to 1.0 and timestamp
        - If new: add with confidence 1.0
        
        Args:
            existing: Current preferences
            new: New preferences to merge
            
        Returns:
            Merged list of preferences
        """
        # Create lookup for existing preferences
        pref_map = {(p.topic, p.verb): p for p in existing}
        
        # Update or add new preferences
        for new_pref in new:
            key = (new_pref.topic, new_pref.verb)
            if key in pref_map:
                # Reset confidence and update timestamp for re-mentioned preference
                pref_map[key].confidence = 1.0
                pref_map[key].last_mentioned = datetime.now()
            else:
                # Add new preference
                pref_map[key] = new_pref
        
        return list(pref_map.values())
    
    def _merge_facts(self, existing: List[Fact], new: List[Fact]) -> List[Fact]:
        """
        Merge new facts with existing ones.
        
        Logic:
        - If fact_type+data already exists: update confidence to 1.0 and timestamp
        - If new: add with confidence 1.0
        
        Args:
            existing: Current facts
            new: New facts to merge
            
        Returns:
            Merged list of facts
        """
        # Create lookup for existing facts (using fact_type + sorted data items as key)
        def fact_key(f: Fact) -> tuple:
            return (f.fact_type, tuple(sorted(f.data.items())))
        
        fact_map = {fact_key(f): f for f in existing}
        
        # Update or add new facts
        for new_fact in new:
            key = fact_key(new_fact)
            if key in fact_map:
                # Reset confidence and update timestamp for re-mentioned fact
                fact_map[key].confidence = 1.0
                fact_map[key].last_mentioned = datetime.now()
            else:
                # Add new fact
                fact_map[key] = new_fact
        
        return list(fact_map.values())
    
    def update(
        self,
        user_id: str,
        sentiments: Optional[List[SentimentResult]] = None,
        preferences: Optional[List[Preference]] = None,
        facts: Optional[List[Fact]] = None
    ) -> UserMemory:
        """
        Update user memory with new extractions.
        
        Process:
        1. Load existing memory
        2. Apply confidence decay to existing items
        3. Merge new items with existing ones
        4. Keep only recent sentiments
        5. Save updated memory
        
        Args:
            user_id: Unique user identifier
            sentiments: New sentiment results to add
            preferences: New preferences to merge
            facts: New facts to merge
            
        Returns:
            Updated UserMemory object
        """
        # Load existing memory
        memory = self.load(user_id)
        
        # Apply decay to existing items
        self._apply_decay(memory)
        
        # Add new sentiments (keep only recent N)
        if sentiments:
            memory.recent_sentiments.extend(sentiments)
            memory.recent_sentiments = memory.recent_sentiments[-self.MAX_RECENT_SENTIMENTS:]
        
        # Merge preferences
        if preferences:
            memory.preferences = self._merge_preferences(memory.preferences, preferences)
        
        # Merge facts
        if facts:
            memory.facts = self._merge_facts(memory.facts, facts)
        
        # Update timestamp
        memory.last_updated = datetime.now()
        
        # Save to disk
        self.save(memory)
        
        return memory
    
    def get_context(
        self, 
        user_id: str, 
        confidence_threshold: float = 0.5
    ) -> str:
        """
        Get formatted memory context for LLM prompts.
        
        Retrieves high-confidence items and formats them into a readable
        string suitable for injection into system prompts.
        
        Args:
            user_id: Unique user identifier
            confidence_threshold: Minimum confidence to include items
            
        Returns:
            Formatted string with user's memory context
        """
        memory = self.load(user_id)
        filtered = memory.get_high_confidence_items(threshold=confidence_threshold)
        
        context_parts = []
        
        # Recent sentiments
        if filtered.recent_sentiments:
            emotions = [f"{s.label} ({s.score:.2f})" for s in filtered.recent_sentiments[-3:]]
            context_parts.append(f"Recent emotions: {', '.join(emotions)}")
        
        # Preferences
        if filtered.preferences:
            prefs = [
                f"{p.verb}s {p.topic} (confidence: {p.confidence:.2f})"
                for p in sorted(filtered.preferences, key=lambda x: x.confidence, reverse=True)[:5]
            ]
            context_parts.append(f"Preferences: {', '.join(prefs)}")
        
        # Facts
        if filtered.facts:
            fact_strings = []
            for f in sorted(filtered.facts, key=lambda x: x.confidence, reverse=True)[:5]:
                if f.fact_type == "relationship":
                    fact_strings.append(f"{f.relation} named {f.person}")
                elif f.fact_type == "location":
                    fact_strings.append(f"{f.context} in {f.place}")
                elif f.fact_type == "life_event":
                    if f.date:
                        fact_strings.append(f"{f.event} in {f.date}")
                    else:
                        fact_strings.append(f"{f.event}")
            
            if fact_strings:
                context_parts.append(f"Facts: {', '.join(fact_strings)}")
        
        if not context_parts:
            return "No significant user history yet."
        
        return " | ".join(context_parts)
    
    def clear(self, user_id: str) -> None:
        """
        Clear all memory for a user.
        
        Args:
            user_id: Unique user identifier
        """
        memory_path = self._get_memory_path(user_id)
        if memory_path.exists():
            memory_path.unlink()


if __name__ == "__main__":
    # Example usage and testing
    store = MemoryStore()
    
    # Create test data
    test_sentiments = [
        SentimentResult(label="joy", score=0.85),
        SentimentResult(label="surprise", score=0.65)
    ]
    
    test_preferences = [
        Preference(topic="coffee", verb="love"),
        Preference(topic="morning", verb="hate")
    ]
    
    test_facts = [
        Fact.from_raw_dict({"type": "relationship", "person": "John", "relation": "friend"}),
        Fact.from_raw_dict({"type": "location", "place": "Seattle", "context": "lives"})
    ]
    
    # Update memory
    print("📝 Updating memory...")
    memory = store.update(
        user_id="test_user",
        sentiments=test_sentiments,
        preferences=test_preferences,
        facts=test_facts
    )
    
    print(f"\n✅ Memory saved with {len(memory.preferences)} preferences and {len(memory.facts)} facts")
    
    # Get context
    print("\n🧠 Memory Context:")
    print(store.get_context("test_user"))
    
    # Simulate second interaction (decay + new data)
    print("\n\n🔄 Second interaction...")
    new_preferences = [Preference(topic="coffee", verb="love")]  # Re-mention coffee
    memory = store.update(user_id="test_user", preferences=new_preferences)
    
    print("\n🧠 Updated Memory Context:")
    print(store.get_context("test_user"))
    
    print(f"\n📊 Confidence levels after decay:")
    for p in memory.preferences:
        print(f"   {p.topic}: {p.confidence:.3f}")
