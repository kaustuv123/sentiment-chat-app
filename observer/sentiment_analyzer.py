import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from config import SENTIMENT_ANALYZER_MODEL

load_dotenv()

# client = InferenceClient(
#     provider="hf-inference",
#     api_key=os.environ["HF_TOKEN"],
# )

# result = client.text_classification(
#     "I like you. I love you",
#     model="j-hartmann/emotion-english-distilroberta-base",
# )
# print(result)

class SentimentAnalyzer:
    def __init__(self, model: str = SENTIMENT_ANALYZER_MODEL):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_TOKEN"],
        )
        self.model = model

    def _analyze(self, text: str):
        """
        Analyzes the sentiment/emotion of the provided text.
        Putting this in a class allows for reusing the InferenceClient instance
        and easily switching models or configurations.
        """
        return self.client.text_classification(text, model=self.model)

    def filter_emotion(self, text: str, threshold: float = 0.30):
        emotions = self._analyze(text)
        selected = []

        for emotion in emotions:
            if emotion['score'] >= threshold:
                selected.append(emotion)

        return selected
