"""
Streamlit UI for Sentiment Chat App

A polished demo interface with:
- Chat interface with memory-aware responses
- Expandable analysis panels per message
- Toggle to switch between memory-enabled and vanilla chatbot modes
"""

import streamlit as st
from orchestrator import ChatOrchestrator
from actor.personality_engine import PersonalityEngine
from actor.gemini_client import GeminiClient


# Page configuration
st.set_page_config(
    page_title="Sentiment Chat",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Analysis card styling */
    .analysis-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }
    
    .analysis-header {
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .sentiment-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .fact-chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #c7d2fe;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0;
    }
    
    .preference-chip {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #a7f3d0;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0;
    }
    
    .personality-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b, #ef4444);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.8);
    }
    
    /* Toggle label */
    .toggle-label {
        font-size: 1rem;
        font-weight: 500;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = ChatOrchestrator(user_id="streamlit_user")
    
    if "vanilla_client" not in st.session_state:
        st.session_state.vanilla_client = GeminiClient()
    
    if "memory_enabled" not in st.session_state:
        st.session_state.memory_enabled = True


def render_analysis_panel(analysis: dict, personality: str):
    """Render the expandable analysis panel for a message."""
    with st.expander("📊 View Analysis", expanded=False):
        cols = st.columns(2)
        
        # Sentiment
        with cols[0]:
            st.markdown("**🎭 Sentiment**")
            sentiments = analysis.get("sentiments", [])
            if sentiments:
                for s in sentiments:
                    if hasattr(s, 'label'):
                        label, score = s.label, s.score
                    else:
                        label, score = s.get("label", "unknown"), s.get("score", 0)
                    st.markdown(f'<span class="sentiment-badge">{label} ({score:.0%})</span>', unsafe_allow_html=True)
                    st.progress(score)
            else:
                st.caption("No strong sentiment detected")
        
        # Personality
        with cols[1]:
            st.markdown("**🎯 Personality**")
            st.markdown(f'<span class="personality-badge">{personality.replace("_", " ").title()}</span>', unsafe_allow_html=True)
        
        # Preferences
        st.markdown("**📋 Preferences Extracted**")
        preferences = analysis.get("preferences", [])
        if preferences:
            chips_html = ""
            for p in preferences:
                if hasattr(p, 'topic'):
                    topic, verb = p.topic, p.verb
                else:
                    topic, verb = p.get("topic", ""), p.get("verb", "")
                chips_html += f'<span class="preference-chip">{verb}s {topic}</span>'
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.caption("None detected")
        
        # Facts
        st.markdown("**📝 Facts Extracted**")
        facts = analysis.get("facts", [])
        if facts:
            chips_html = ""
            for f in facts:
                if hasattr(f, 'fact_type'):
                    fact_type = f.fact_type
                    if fact_type == "relationship":
                        chips_html += f'<span class="fact-chip">{f.relation}: {f.person}</span>'
                    elif fact_type == "location":
                        chips_html += f'<span class="fact-chip">{f.context} in {f.place}</span>'
                    elif fact_type == "life_event":
                        chips_html += f'<span class="fact-chip">{f.event}</span>'
                else:
                    fact_type = f.get("type", "unknown")
                    if fact_type == "relationship":
                        chips_html += f'<span class="fact-chip">{f.get("relation", "")}: {f.get("person", "")}</span>'
                    elif fact_type == "location":
                        chips_html += f'<span class="fact-chip">{f.get("context", "")} in {f.get("place", "")}</span>'
                    elif fact_type == "life_event":
                        chips_html += f'<span class="fact-chip">{f.get("event", "")}</span>'
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.caption("None detected")


def main():
    """Main app entry point."""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🧠 Sentiment Chat")
        st.markdown("---")
        
        # Memory toggle
        st.session_state.memory_enabled = st.toggle(
            "Enable Memory & Analysis",
            value=st.session_state.memory_enabled,
            help="When ON, the bot analyzes your messages and remembers context. When OFF, it's a vanilla chatbot."
        )
        
        if st.session_state.memory_enabled:
            st.success("🟢 Memory Active")
            st.caption("Analyzing sentiments, preferences, and facts")
        else:
            st.info("⚪ Vanilla Mode")
            st.caption("Standard chatbot without memory")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.orchestrator = ChatOrchestrator(user_id="streamlit_user")
            st.session_state.vanilla_client = GeminiClient()
            st.rerun()
        
        # Memory summary (only when memory is enabled)
        if st.session_state.memory_enabled:
            st.markdown("---")
            st.markdown("### 🧠 Memory Summary")
            memory_summary = st.session_state.orchestrator.get_memory_summary()
            if memory_summary != "No significant user history yet.":
                st.caption(memory_summary)
            else:
                st.caption("No memories yet. Start chatting!")
    
    # Main chat area
    st.markdown("# 💬 Chat")
    
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show analysis panel for assistant messages (only if memory was enabled)
            if msg["role"] == "assistant" and msg.get("analysis"):
                render_analysis_panel(msg["analysis"], msg.get("personality", "unknown"))
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.memory_enabled:
                    # Full pipeline with memory
                    result = st.session_state.orchestrator.process_message(prompt)
                    response = result["response"]
                    analysis = result["analysis"]
                    personality = result["personality"]
                else:
                    # Vanilla mode - no memory
                    if st.session_state.vanilla_client.chat is None:
                        st.session_state.vanilla_client.start_chat(
                            PersonalityEngine.get_neutral_prompt()
                        )
                    response = st.session_state.vanilla_client.send_message(prompt)
                    analysis = None
                    personality = None
            
            st.markdown(response)
            
            # Show analysis panel if memory enabled
            if st.session_state.memory_enabled and analysis:
                render_analysis_panel(analysis, personality)
        
        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "analysis": analysis,
            "personality": personality
        })


if __name__ == "__main__":
    main()
