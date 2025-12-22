import spacy
from typing import List, Dict

class FactExtractor:
    """
    Extracts memorable facts from text using SpaCy NER + dependency parsing.
    
    Fact Types:
    - Relationships: "my friend John" → {type: "relationship", person: "John", relation: "friend"}
    - Locations: "lives in Seattle" → {type: "location", place: "Seattle", context: "lives"}
    - Life Events: "graduated in 2020" → {type: "life_event", event: "graduated", date: "2020"}
    """
    
    RELATIONSHIP_TRIGGERS = ["wife", "husband", "friend", "brother", "sister", "boss", 
                            "mother", "father", "son", "daughter", "colleague", "partner"]
    
    LIFE_EVENT_VERBS = ["married", "divorced", "graduated", "moved", "started", "quit",
                        "born", "hired", "retired", "promoted"]
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        facts = []
        
        # Extract relationships (PERSON entities with relationship context)
        facts.extend(self._extract_relationships(doc))
        
        # Extract locations (GPE entities with context)
        facts.extend(self._extract_locations(doc))
        
        # Extract life events (verbs + dates)
        facts.extend(self._extract_life_events(doc))
        
        return facts
    
    def _extract_relationships(self, doc) -> List[Dict]:
        """Find patterns like 'my friend John' where PERSON has appos pointing to relationship"""
        relationships = []
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check if the person name is in apposition to a relationship word
                for token in ent:
                    if token.dep_ == "appos" and token.head.text.lower() in self.RELATIONSHIP_TRIGGERS:
                        relationships.append({
                            "type": "relationship",
                            "person": ent.text,
                            "relation": token.head.text.lower()
                        })
                        break
                    # Also check if the person's head is a relationship trigger
                    elif token.head.text.lower() in self.RELATIONSHIP_TRIGGERS:
                        relationships.append({
                            "type": "relationship", 
                            "person": ent.text,
                            "relation": token.head.text.lower()
                        })
                        break
        
        return relationships
    
    def _extract_locations(self, doc) -> List[Dict]:
        """Find patterns like 'lives in Seattle' - GPE entities with verb context"""
        locations = []
        
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Geo-Political Entity (cities, countries, etc.)
                # Traverse up to find the verb (location → prep → verb)
                for token in ent:
                    if token.dep_ == "pobj":  # object of preposition
                        prep = token.head  # "in"
                        if prep.dep_ == "prep":
                            verb = prep.head  # "lives"
                            locations.append({
                                "type": "location",
                                "place": ent.text,
                                "context": verb.text
                            })
                            break
        
        return locations
    
    def _extract_life_events(self, doc) -> List[Dict]:
        """Find patterns like 'graduated in 2020' - life event verb + DATE"""
        events = []
        
        for token in doc:
            if token.lemma_.lower() in self.LIFE_EVENT_VERBS:
                # Look for associated DATE entity
                event_date = None
                
                # Check children for prep phrases containing dates
                for child in token.children:
                    if child.dep_ == "prep":  # "in"
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                # Check if this is a date entity
                                for ent in doc.ents:
                                    if ent.label_ == "DATE" and grandchild in ent:
                                        event_date = ent.text
                                        break
                
                events.append({
                    "type": "life_event",
                    "event": token.lemma_.lower(),
                    "date": event_date  # Could be None if no date found
                })
        
        return events


if __name__ == "__main__":
    extractor = FactExtractor()
    
    # Test cases
    test_sentences = [
        "My friend John lives in Seattle and graduated in 2020",
        "My sister Lisa moved to New York",
        "I got married in 2018",
    ]
    
    for sentence in test_sentences:
        print(f"\n📝 '{sentence}'")
        facts = extractor.extract(sentence)
        for fact in facts:
            print(f"   → {fact}")