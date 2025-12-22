import spacy
from typing import List

class PreferenceExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str):
        doc = self.nlp(text)
        preferences = []
        
        # Check for multiple dependency types that indicate preference objects
        valid_deps = ["dobj", "npadvmod", "pobj"]
        
        for token in doc:
            if token.dep_ in valid_deps:
                topic = token.text
                verb = token.head.text
                preferences.append({"topic": topic, "verb": verb})
        
        return preferences

if __name__ == "__main__":
    extractor = PreferenceExtractor()
    print(extractor.extract("I hate the morning"))