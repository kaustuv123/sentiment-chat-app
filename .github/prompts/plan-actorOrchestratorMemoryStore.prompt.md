## Plan: Build Actor, Orchestrator & Memory Store

This plan adds the remaining infrastructure to complete the Sentiment Chat App: a **Memory Store** for persisting observer extractions with confidence decay, an **Actor Layer** (personality engine + Gemini client) for generating persona-driven responses, and an **Orchestrator** to coordinate everything.

### Steps

1. **Create data models** in `observer/__init__.py` — define `SentimentResult`, `Preference`, `Fact`, and `UserMemory` dataclasses to standardize return types across extractors and memory (currently using raw dicts).
`PreferenceExtractor` and `FactExtractor` load `en_core_web_sm` separately — create a shared NLP singleton in the orchestrator to reduce memory usage.

2. **Implement memory store** at `observer/memory_store.py` — create `MemoryStore` class with JSON persistence to `data/memory/`, confidence decay logic (`DECAY_RATE = 0.95`), `update()` method to merge new extractions, and `get_context()` to retrieve formatted memory for prompts.

3. **Build personality engine** at `actor/personality_engine.py` — define 5 personality system prompt templates (Calm Mentor, Witty Friend, Therapist, Motivational Coach, Sarcastic Buddy) with a `get_system_prompt(personality, memory)` method that injects relevant user memory context.

4. **Create Gemini client** at `actor/gemini_client.py` — wrapper using `google-generativeai` SDK with `generate_response(message, system_prompt, history)` method; add `GEMINI_API_KEY` to `config.py`.

5. **Implement orchestrator** at `orchestrator.py` — create `ChatOrchestrator` class that coordinates: observer analysis → memory update → actor response generation; expose `process_message(text, personality)` returning both neutral and personality-adjusted responses.

6. **Update dependencies** in `requirements.txt` — add `google-generativeai>=0.3.0`, `pydantic>=2.5.0`, `streamlit>=1.28.0`.

