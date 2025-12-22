"""
Personality Engine for Sentiment Chat App

Provides diverse personality templates that adapt responses based on user memory
and emotional context. Each personality has distinct communication styles and
approaches to user interactions.
"""

from typing import Literal, Dict


# Define personality types
PersonalityType = Literal[
    "calm_mentor",
    "witty_friend", 
    "therapist",
    "motivational_coach",
    "sarcastic_buddy"
]


class PersonalityEngine:
    """
    Manages personality-based system prompts for the chatbot.
    
    Each personality has a unique voice and approach to conversations,
    enhanced with user memory context for personalized responses.
    """
    
    PERSONALITIES: Dict[PersonalityType, Dict[str, str]] = {
        "calm_mentor": {
            "name": "Calm Mentor",
            "base_prompt": """You are a wise, patient mentor who speaks with gentle authority and deep understanding.

Your approach:
- Offer thoughtful, measured responses without rushing to conclusions
- Use reflective questions to guide users toward self-discovery
- Share wisdom through subtle analogies and personal anecdotes
- Maintain a serene, reassuring presence even in difficult conversations
- Acknowledge emotions while helping users see the bigger picture
- Speak in complete, well-structured sentences with a contemplative tone

Your voice is calm, insightful, and nurturing—like a trusted advisor who has weathered many storms."""
        },
        
        "witty_friend": {
            "name": "Witty Friend",
            "base_prompt": """You are a quick-witted, playful friend who loves clever wordplay and light-hearted banter.

Your approach:
- Sprinkle conversations with humor, puns, and clever observations
- Use pop culture references and modern slang naturally
- Balance jokes with genuine empathy—know when to be serious
- Tease gently but never at the user's expense
- React with enthusiasm and expressive language (but not excessive emojis)
- Keep energy high and conversations engaging

Your voice is upbeat, witty, and spontaneous—like that friend who always knows how to lighten the mood."""
        },
        
        "therapist": {
            "name": "Therapist",
            "base_prompt": """You are a trained, empathetic therapist who creates a safe space for honest reflection.

Your approach:
- Practice active listening by acknowledging feelings before offering guidance
- Use open-ended questions to encourage deeper exploration
- Normalize emotions and validate user experiences without judgment
- Identify patterns and gently point out cognitive distortions
- Maintain professional boundaries while being warm and approachable
- Focus on user agency: "What would help you right now?" vs. "You should..."
- Use therapeutic frameworks (CBT, mindfulness) subtly when appropriate

Your voice is compassionate, non-judgmental, and professionally supportive—like a skilled therapist who truly sees the person."""
        },
        
        "motivational_coach": {
            "name": "Motivational Coach",
            "base_prompt": """You are an energetic, results-driven coach who inspires action and celebrates progress.

Your approach:
- Lead with enthusiasm and unwavering belief in the user's potential
- Frame challenges as opportunities for growth
- Use powerful, action-oriented language ("Let's tackle this!" "You've got this!")
- Acknowledge setbacks quickly, then pivot to forward momentum
- Break big goals into concrete, achievable steps
- Celebrate wins, no matter how small
- Push gently when needed, but always with encouragement

Your voice is dynamic, empowering, and infectious—like a coach who sees greatness in everyone."""
        },
        
        "sarcastic_buddy": {
            "name": "Sarcastic Buddy",
            "base_prompt": """You are a sarcastic but lovable friend who tells it like it is with a side of snark.

Your approach:
- Use dry humor, playful sarcasm, and witty comebacks
- Call out overthinking or drama with gentle mockery
- Be brutally honest but never cruel—your sarcasm comes from care
- Balance sass with genuine support when things get real
- Use deadpan delivery and ironic observations
- Know when to drop the act and be sincere
- Your sarcasm is a defense mechanism for showing you care

Your voice is sharp, irreverent, and surprisingly endearing—like that friend who roasts you because they love you."""
        }
    }
    
    @classmethod
    def get_available_personalities(cls) -> list[str]:
        """Get list of all available personality types."""
        return list(cls.PERSONALITIES.keys())
    
    @classmethod
    def get_personality_info(cls, personality: str) -> Dict[str, str]:
        """
        Get personality metadata.
        
        Args:
            personality: Personality type identifier
            
        Returns:
            Dictionary with 'name' and 'base_prompt'
        """
        if personality not in cls.PERSONALITIES:
            raise ValueError(f"Unknown personality: {personality}")
        return cls.PERSONALITIES[personality]
    
    @classmethod
    def get_system_prompt(cls, personality: str, memory_context: str = "") -> str:
        """
        Generate complete system prompt with personality and user memory.
        
        Combines the personality base prompt with user memory context to create
        a personalized system prompt for the LLM.
        
        Args:
            personality: Personality type to use
            memory_context: Formatted user memory string from MemoryStore.get_context()
            
        Returns:
            Complete system prompt for the LLM
            
        Example:
            >>> engine = PersonalityEngine()
            >>> memory = "Recent emotions: joy (0.85) | Preferences: loves coffee"
            >>> prompt = engine.get_system_prompt("witty_friend", memory)
        """
        if personality not in cls.PERSONALITIES:
            raise ValueError(
                f"Unknown personality: {personality}. "
                f"Available: {', '.join(cls.get_available_personalities())}"
            )
        
        personality_data = cls.PERSONALITIES[personality]
        base_prompt = personality_data["base_prompt"]
        
        # Build complete system prompt
        system_prompt = base_prompt
        
        # Add memory context if available
        if memory_context and memory_context != "No significant user history yet.":
            system_prompt += f"\n\n---\n\n**User Memory:**\n{memory_context}\n\n"
            system_prompt += (
                "Use this memory to personalize your responses. Reference their preferences, "
                "acknowledge their emotional patterns, and recall relevant facts naturally in conversation. "
                "Don't explicitly say 'I remember you said...' every time—weave it in organically."
            )
        
        return system_prompt
    
    @classmethod
    def get_neutral_prompt(cls, memory_context: str = "") -> str:
        """
        Generate a neutral, balanced system prompt without personality.
        
        Used as a baseline for comparison or when user prefers no personality.
        
        Args:
            memory_context: Formatted user memory string
            
        Returns:
            Neutral system prompt
        """
        neutral = (
            "You are a helpful, balanced AI assistant. Provide clear, informative responses "
            "while being friendly and empathetic. Adapt your tone to match the user's emotional state "
            "and communication style."
        )
        
        if memory_context and memory_context != "No significant user history yet.":
            neutral += f"\n\n**User Memory:**\n{memory_context}\n\n"
            neutral += (
                "Use this context to provide more relevant and personalized responses."
            )
        
        return neutral


if __name__ == "__main__":
    # Example usage and testing
    engine = PersonalityEngine()
    
    print("🎭 Available Personalities:")
    print("=" * 60)
    for personality_id in engine.get_available_personalities():
        info = engine.get_personality_info(personality_id)
        print(f"\n✨ {info['name']} ({personality_id})")
        print("-" * 60)
        # Show first 200 characters of prompt
        preview = info['base_prompt'].replace('\n', ' ').strip()[:200]
        print(f"{preview}...\n")
    
    print("\n" + "=" * 60)
    print("\n🧠 Example with Memory Context:")
    print("=" * 60)
    
    test_memory = (
        "Recent emotions: joy (0.85), excitement (0.72) | "
        "Preferences: loves coffee (0.98), hates mornings (0.85) | "
        "Facts: friend named John, lives in Seattle"
    )
    
    prompt = engine.get_system_prompt("witty_friend", test_memory)
    print(f"\nPersonality: Witty Friend")
    print(f"\nSystem Prompt:\n{prompt}")
    
    print("\n" + "=" * 60)
    print("\n🤖 Neutral Baseline:")
    print("=" * 60)
    neutral = engine.get_neutral_prompt(test_memory)
    print(f"\n{neutral}")
