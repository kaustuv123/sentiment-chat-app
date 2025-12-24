"""
Chat Orchestrator - Coordinates Observer → Memory → Actor pipeline

This is the central hub that:
1. Analyzes user messages with observer components (sentiment, facts, preferences)
2. Updates persistent memory with extracted knowledge
3. Generates personality-driven responses using Gemini
"""

from observer.sentiment_analyzer import SentimentAnalyzer
from observer.fact_extractor import FactExtractor
from observer.preference_extractor import PreferenceExtractor
from observer.memory_store import MemoryStore
from observer.models import SentimentResult, Preference, Fact

from actor.gemini_client import GeminiClient
from actor.personality_engine import PersonalityEngine
from config import GEMINI_MODEL
from google import genai
from google.genai import types
import os


class ChatOrchestrator:
    """
    Coordinates the full pipeline:
    User Message → Observer Analysis → Memory Update → Actor Response
    
    Usage:
        orch = ChatOrchestrator(user_id="user123")
        result = orch.process_message("I love coffee!", personality="witty_friend")
        print(result["response"])
    """
    
    def __init__(self, user_id: str = "default_user"):
        """
        Initialize all components.
        
        Args:
            user_id: Unique identifier for the user (used for memory persistence)
        """
        self.user_id = user_id
        
        # Observer components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.fact_extractor = FactExtractor()
        self.preference_extractor = PreferenceExtractor()
        
        # Memory
        self.memory_store = MemoryStore()
        
        # Actor components
        self.gemini_client = GeminiClient()
        self.current_personality = None
        
        # Personality selector client (separate from chat)
        self.selector_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Prompt for personality selection
    PERSONALITY_SELECTOR_PROMPT = """You are a personality selector for a chatbot. Based on the user's message and their memory context (past emotions, preferences, facts), select the BEST personality to respond with.

Available personalities:
- calm_mentor: For users needing guidance, wisdom, or who seem confused/overwhelmed
- witty_friend: For users in good mood, casual chat, or who enjoy humor
- therapist: For users expressing sadness, anxiety, fear, or emotional struggles  
- motivational_coach: For users who need encouragement, are frustrated, or working on goals
- sarcastic_buddy: For users who enjoy playful banter or need tough love

Respond with ONLY the personality ID (e.g., "therapist"), nothing else."""

    def _select_personality(self, message: str, memory_context: str) -> str:
        """
        Use Gemini to select the best personality based on message and memory.
        
        Args:
            message: Current user message
            memory_context: Formatted memory string from MemoryStore
            
        Returns:
            Personality ID string (e.g., "witty_friend")
        """
        prompt = f"""User's message: "{message}"

User's memory context: {memory_context}

Based on this, which personality should respond?"""

        try:
            response = self.selector_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.PERSONALITY_SELECTOR_PROMPT,
                    temperature=0.3,  # Low temperature for consistent selection
                    max_output_tokens=20,
                )
            )
            
            selected = response.text.strip().lower().replace('"', '').replace("'", "").replace(".", "")
            
            # Validate selection - direct match
            valid_personalities = PersonalityEngine.get_available_personalities()
            if selected in valid_personalities:
                return selected
            
            # Partial match - find personality that starts with the response
            for personality in valid_personalities:
                if personality.startswith(selected) or selected.startswith(personality.split("_")[0]):
                    print(f"🔧 Matched partial '{selected}' → '{personality}'")
                    return personality
            
            # Keyword match fallback
            keyword_map = {
                "witty": "witty_friend", "friend": "witty_friend", "w": "witty_friend",
                "calm": "calm_mentor", "mentor": "calm_mentor", "c": "calm_mentor",
                "therapist": "therapist", "therapy": "therapist", "t": "therapist",
                "coach": "motivational_coach", "motiv": "motivational_coach", "m": "motivational_coach",
                "sarcas": "sarcastic_buddy", "buddy": "sarcastic_buddy", "s": "sarcastic_buddy",
            }
            for keyword, personality in keyword_map.items():
                if keyword in selected or selected == keyword:
                    print(f"🔧 Keyword matched '{selected}' → '{personality}'")
                    return personality
            
            print(f"⚠️ Could not parse '{selected}', defaulting to calm_mentor")
            return "calm_mentor"
                
        except Exception as e:
            print(f"⚠️ Personality selection failed: {e}, defaulting to calm_mentor")
            return "calm_mentor"
    
    def _analyze_message(self, text: str) -> dict:
        """
        Run all observer extractions on the message.
        
        Args:
            text: User's message
            
        Returns:
            Dictionary with sentiments, facts, and preferences
        """
        return {
            "sentiments": self.sentiment_analyzer.filter_emotion(text),
            "facts": self.fact_extractor.extract(text),
            "preferences": self.preference_extractor.extract(text)
        }
    
    def _convert_to_models(self, analysis: dict) -> dict:
        """
        Convert raw extractor dicts to proper model objects.
        
        Args:
            analysis: Raw extraction results
            
        Returns:
            Dictionary with properly typed model objects
        """
        # Convert sentiments: {"label": "joy", "score": 0.85} → SentimentResult
        sentiments = [
            SentimentResult(label=s["label"], score=s["score"])
            for s in analysis.get("sentiments", [])
        ]
        
        # Convert preferences: {"topic": "coffee", "verb": "love"} → Preference
        preferences = [
            Preference(topic=p["topic"], verb=p["verb"])
            for p in analysis.get("preferences", [])
        ]
        
        # Convert facts: {"type": "relationship", ...} → Fact
        facts = [
            Fact.from_raw_dict(f)
            for f in analysis.get("facts", [])
        ]
        
        return {
            "sentiments": sentiments,
            "preferences": preferences,
            "facts": facts
        }
    
    def _update_memory(self, models: dict) -> None:
        """
        Update memory store with extracted model objects.
        
        Args:
            models: Dictionary with SentimentResult, Preference, Fact lists
        """
        self.memory_store.update(
            user_id=self.user_id,
            sentiments=models.get("sentiments"),
            preferences=models.get("preferences"),
            facts=models.get("facts")
        )
    
    def process_message(
        self, 
        message: str, 
        personality: str = "auto"
    ) -> dict:
        """
        Main entry point - process user message and return response.
        
        Pipeline:
        1. Analyze message with observers (sentiment, facts, preferences)
        2. Convert raw results to model objects
        3. Update memory with extracted knowledge
        4. Get memory context for personalization
        5. Select personality using Gemini (if auto)
        6. Build system prompt with personality + memory
        7. Send to Gemini and get response
        
        Args:
            message: User's input message
            personality: Personality type or "auto" for Gemini to select
                        Options: calm_mentor, witty_friend, therapist, 
                        motivational_coach, sarcastic_buddy, auto
        
        Returns:
            {
                "response": "AI response text",
                "personality": "selected personality",
                "analysis": {...},
                "memory_context": "..."
            }
        """
        # Step 1: Analyze message with observers
        raw_analysis = self._analyze_message(message)
        
        # Step 2: Convert to model objects
        models = self._convert_to_models(raw_analysis)
        
        # Step 3: Update memory
        self._update_memory(models)
        
        # Step 4: Get memory context
        memory_context = self.memory_store.get_context(self.user_id)
        
        # Step 5: Select personality (auto or manual)
        if personality == "auto":
            personality = self._select_personality(message, memory_context)
            print(f"🎭 Auto-selected personality: {personality}")
        
        # Step 6: Build system prompt with personality + memory
        system_prompt = PersonalityEngine.get_system_prompt(
            personality=personality,
            memory_context=memory_context
        )
        
        # Step 7: Start chat (if personality changed) and send message
        if self.current_personality != personality:
            self.gemini_client.start_chat(system_prompt)
            self.current_personality = personality
        
        response = self.gemini_client.send_message(message)
        
        return {
            "response": response,
            "personality": personality,
            "analysis": raw_analysis,
            "memory_context": memory_context
        }
    
    def reset_conversation(self, personality: str = None) -> None:
        """
        Reset the chat session (clears Gemini history, keeps memory).
        
        Args:
            personality: Optional new personality to use
        """
        new_personality = personality or self.current_personality or "calm_mentor"
        memory_context = self.memory_store.get_context(self.user_id)
        system_prompt = PersonalityEngine.get_system_prompt(new_personality, memory_context)
        
        self.gemini_client.start_chat(system_prompt)
        self.current_personality = new_personality
    
    def clear_memory(self) -> None:
        """Clear all memory for this user."""
        self.memory_store.clear(self.user_id)
    
    def get_memory_summary(self) -> str:
        """Get current memory context as string."""
        return self.memory_store.get_context(self.user_id)


if __name__ == "__main__":
    print("🚀 Testing ChatOrchestrator with Auto Personality Selection...\n")
    print("=" * 60)
    
    # Initialize orchestrator
    orch = ChatOrchestrator(user_id="test_user")
    
    # Test conversation - personality will be auto-selected
    test_messages = [
        "Hi! I'm excited about my new job in Seattle!",
        "I've been feeling really anxious about my performance though.",
        "My friend John keeps telling me to just relax.",
    ]
    
    for message in test_messages:
        print(f"\n👤 You: {message}")
        print("-" * 40)
        
        result = orch.process_message(message)  # Auto personality selection!
        
        print(f"🎭 Personality: {result['personality']}")
        print(f"🤖 Bot: {result['response']}")
        print(f"\n📊 Analysis:")
        print(f"   Sentiments: {result['analysis']['sentiments']}")
        print(f"   Preferences: {result['analysis']['preferences']}")
        print(f"   Facts: {result['analysis']['facts']}")
        print(f"\n🧠 Memory: {result['memory_context']}")
        print("=" * 60)
    
    print("\n✅ Orchestrator test complete!")
