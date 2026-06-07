"""
Streamlit frontend for the Podcast Q&A Bot.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from chatbot import PodcastQA


# Cache the heavy objects (embedding model + FAISS index + LLM client) so
# they are only loaded once per Streamlit server session.
@st.cache_resource(show_spinner="Loading index, embedding model, and LLM client...")
def load_qa() -> PodcastQA:
    return PodcastQA()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Podcast Q&A Bot | People by WTF Ep. 16",
    page_icon="🎙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .card {
        background-color: #1e1e1e;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #2e2e2e;
        margin-bottom: 1rem;
        color: #f5f5f5;
    }
    .card h4 {
        margin: 0 0 0.5rem 0;
        color: #ff4b4b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .card p { margin: 0; line-height: 1.55; white-space: pre-wrap; }
    .answer-card { border-left: 4px solid #ff4b4b; }
    .ts-card     { border-left: 4px solid #4b8bff; }
    .tr-card     { border-left: 4px solid #6fcf97; }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #ff6b6b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎙️ Podcast Q&A Bot")
st.caption(
    "Ask anything about *Elon Musk × Nikhil Kamath | People by WTF Ep. 16*. "
    "Answers are grounded in the actual podcast transcript with deep-links "
    "to the exact moment they were discussed."
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "RAG chatbot over a single podcast episode. Audio is transcribed with "
        "Whisper, embedded with Sentence-Transformers, indexed in FAISS, and "
        "answered by an OpenAI chat model."
    )
    st.markdown("### Pipeline")
    st.markdown(
        """
        1. **YouTube → Audio** (`yt-dlp`)
        2. **Audio → Transcript** (`Whisper`)
        3. **Transcript → Chunks** (timestamp-preserving)
        4. **Chunks → Embeddings** (`Sentence-Transformers`)
        5. **Embeddings → FAISS** index
        6. **Question → Top-k → LLM → Answer + Timestamp**
        """
    )
    st.markdown("### Tips")
    st.write("Ask specific questions for the best results.")

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------
question = st.text_input(
    "Ask a question about the podcast",
    placeholder="e.g. What does Elon Musk say about first-principles thinking?",
    label_visibility="collapsed",
)

col_ask, col_example, _ = st.columns([1, 2, 5])
ask_clicked = col_ask.button("Ask", type="primary")
example_clicked = col_example.button("Try: What is first-principles thinking?")

if example_clicked:
    question = "What is first-principles thinking?"

if ask_clicked or example_clicked:
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    with st.spinner("Searching the transcript and drafting an answer..."):
        try:
            qa = load_qa()
            result = qa.ask(question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error: {exc}")
            st.stop()

    # ---------------------------------- Result --------------------------------
    st.subheader("Result")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="card answer-card">'
            f'<h4>Answer</h4><p>{result.answer}</p></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="card ts-card">'
            f'<h4>Timestamp</h4><p>{result.timestamp}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="card tr-card">'
        f'<h4>Transcript Segment</h4><p>{result.transcript}</p></div>',
        unsafe_allow_html=True,
    )

    st.link_button(
        "▶  Open Video at this moment",
        result.youtube_url,
        use_container_width=False,
    )

    with st.expander("Show retrieved context (top-k chunks)"):
        for i, c in enumerate(result.chunks, 1):
            st.markdown(
                f"**{i}. {c['start']:.1f}s → {c['end']:.1f}s "
                f"(score: {c['score']:.3f})**\n\n{c['text']}"
            )

    st.caption(
        "Note: the LLM is instructed to refuse to answer if the context "
        "doesn't contain the information."
    )
