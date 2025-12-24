import os
from dotenv import load_dotenv
from config import GEMINI_API_KEY, GEMINI_MODEL
from google import genai
from google.genai import types
from typing import Optional

load_dotenv()


class GeminiClient:
    """
    Wrapper for Google Gemini API with multi-turn chat support.
    
    Usage:
        client = GeminiClient()
        client.start_chat("You are a helpful assistant.")
        response = client.send_message("Hello!")
    """
    
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = GEMINI_MODEL
        self.chat = None  # Will be created when start_chat() is called
        self.current_system_prompt = None
    
    def start_chat(self, system_prompt: str = None) -> None:
        """
        Start a new chat session with optional system prompt.
        
        Args:
            system_prompt: Personality/context instructions for the model
        """
        self.current_system_prompt = system_prompt
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=1000,
        ) if system_prompt else None
        
        self.chat = self.client.chats.create(
            model=self.model,
            config=config
        )
    
    def send_message(self, message: str, stream: bool = False) -> str:
        """
        Send a message in the current chat session.
        
        Args:
            message: User's message
            stream: If True, returns a generator for streaming
            
        Returns:
            Model's response text
        """
        if self.chat is None:
            self.start_chat()
        
        if stream:
            response = self.chat.send_message_stream(message)
            return response  # Returns generator, caller handles iteration
        else:
            response = self.chat.send_message(message)
            return response.text
    
    def get_history(self) -> list:
        """Get conversation history from current chat."""
        if self.chat:
            return self.chat.get_history()
        return []
    
    def reset_chat(self, system_prompt: str = None) -> None:
        """Reset chat with new or same system prompt."""
        prompt = system_prompt or self.current_system_prompt
        self.start_chat(prompt)


# Test code - only runs when executed directly
if __name__ == "__main__":
    print("Testing GeminiClient...")
    
    client = GeminiClient()
    client.start_chat("You are a witty friend who loves puns.")
    
    # Test multi-turn conversation
    response1 = client.send_message("Hi! I'm learning Python.")
    print(f"Bot: {response1}\n")
    
    response2 = client.send_message("What's the best thing about it?")
    print(f"Bot: {response2}\n")
    
    # Show history
    print("--- Conversation History ---")
    for msg in client.get_history():
        print(f"{msg.role}: {msg.parts[0].text[:100]}...")