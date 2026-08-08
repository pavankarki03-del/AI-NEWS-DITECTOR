"""
app.py
News Topic Detector – Streamlit app supporting Ollama, Sentence Transformers, and Google Gemini
"""

import os

# Suppress HuggingFace Hub symlink warning on Windows & Tokenizers warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="News Topic Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 News Topic Detector")
st.markdown("Paste any news headline or article to automatically detect its primary topic.")

# Common Topic Taxonomy
TOPIC_LIST = [
    "Politics", "Business & Economy", "Technology", "Science",
    "Health", "Sports", "Entertainment", "Environment",
    "Crime & Law", "Education", "World Affairs", "Other"
]

# ---------- Helper Functions ----------

@st.cache_resource
def load_sentence_transformer():
    """Load local SentenceTransformer model cached for speed."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

def get_ollama_models():
    """Safely fetch available Ollama models handling ListResponse objects & dicts."""
    try:
        import ollama
        response = ollama.list()
        if hasattr(response, "models"):
            models_list = response.models
        elif isinstance(response, dict):
            models_list = response.get("models", [])
        else:
            models_list = []
        
        extracted = []
        for m in models_list:
            if hasattr(m, "model"):
                name = m.model or getattr(m, "name", None)
            elif isinstance(m, dict):
                name = m.get("name") or m.get("model")
            else:
                name = str(m)
            if name:
                extracted.append(name)
        return extracted
    except Exception:
        return []

def classify_with_ollama(text, model_name, temp):
    """Classify topic using local Ollama instance."""
    import ollama
    prompt = f"""You are a news topic classifier.
Analyze the following news text and return ONLY the single most relevant topic from this list:

{', '.join(TOPIC_LIST)}

News text:
\"\"\"{text}\"\"\"

Respond with just the topic name, nothing else."""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temp}
    )

    # Handle ChatResponse object vs dict
    if hasattr(response, "message") and hasattr(response.message, "content"):
        content = response.message.content
    elif isinstance(response, dict):
        content = response.get("message", {}).get("content", "")
    else:
        content = str(response)

    return content.strip(), content

def classify_with_sentence_transformers(text):
    """Classify topic using Sentence Transformers cosine similarity."""
    from sentence_transformers import util
    model = load_sentence_transformer()
    
    text_emb = model.encode(text, convert_to_tensor=True)
    topic_embs = model.encode(TOPIC_LIST, convert_to_tensor=True)
    
    scores = util.cos_sim(text_emb, topic_embs)[0].tolist()
    
    # Pair topics with scores
    topic_scores = sorted(zip(TOPIC_LIST, scores), key=lambda x: x[1], reverse=True)
    top_topic, top_score = topic_scores[0]
    return top_topic, topic_scores

def classify_with_gemini(text, api_key):
    """Classify topic using Google Gemini API."""
    from google import genai
    client = genai.Client(api_key=api_key)
    prompt = f"""You are a news topic classifier.
Analyze the following news text and return ONLY the single most relevant topic from this list:

{', '.join(TOPIC_LIST)}

News text:
\"\"\"{text}\"\"\"

Respond with just the topic name, nothing else."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip(), response.text

# ---------- Sidebar Settings ----------
with st.sidebar:
    st.header("⚙️ Model Settings")
    
    provider = st.radio(
        "Select Provider",
        ["Sentence Transformers (Local Vector)", "Ollama (Local LLM)", "Google Gemini (Cloud AI)"],
        help="Choose the model engine for topic detection."
    )
    
    selected_model = None
    temperature = 0.3
    gemini_key = os.getenv("GOOGLE_API_KEY", "")

    if provider == "Ollama (Local LLM)":
        ollama_models = get_ollama_models()
        if not ollama_models:
            st.warning("⚠️ No running Ollama instance or models found.\n\nMake sure Ollama is running (`ollama serve`) and a model is pulled (`ollama pull llama3.2`).")
            selected_model = st.text_input("Enter Ollama Model Name", value="llama3.2")
        else:
            selected_model = st.selectbox("Select Ollama Model", ollama_models)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)

    elif provider == "Google Gemini (Cloud AI)":
        gemini_key = st.text_input("Google API Key", value=gemini_key, type="password")
        if not gemini_key:
            st.info("Provide a Google API key or set `GOOGLE_API_KEY` in `.env`.")

    elif provider == "Sentence Transformers (Local Vector)":
        st.success("✅ Ready (Fast, lightweight local embedding model)")

# ---------- Sample Buttons ----------
st.subheader("Try a Sample Headline:")
col1, col2, col3, col4 = st.columns(4)
sample_text = ""
if col1.button("📈 Stock Market"):
    sample_text = "Global stock markets rally as tech shares jump following impressive quarterly earnings reports."
if col2.button("🚀 Space Discovery"):
    sample_text = "Astronomers discover water vapor on an Earth-sized exoplanet orbiting a nearby star."
if col3.button("⚽ Championship"):
    sample_text = "Real Madrid secures dramatic late victory in Champions League final with injury-time goal."
if col4.button("🌱 Climate Summit"):
    sample_text = "World leaders gather in Geneva to sign historic accord on reducing carbon emissions by 2030."

# ---------- Main Input ----------
news_text = st.text_area(
    "News Article or Headline",
    value=sample_text,
    height=150,
    placeholder="Paste news text here..."
)

# ---------- Detect Button & Action ----------
if st.button("Detect Topic 🔍", type="primary"):
    if not news_text.strip():
        st.warning("Please enter or select news text first.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                if provider == "Sentence Transformers (Local Vector)":
                    topic, scores = classify_with_sentence_transformers(news_text)
                    st.success(f"**Detected Topic:** {topic}")
                    
                    # Display top topic confidence breakdown
                    st.subheader("Topic Confidence Scores:")
                    for top_t, score in scores[:5]:
                        st.write(f"**{top_t}**: {max(0.0, score):.2%}")
                        st.progress(max(0.0, min(1.0, score)))

                elif provider == "Ollama (Local LLM)":
                    topic, raw_resp = classify_with_ollama(news_text, selected_model, temperature)
                    st.success(f"**Detected Topic:** {topic}")
                    with st.expander("Show Raw Model Response"):
                        st.write(raw_resp)

                elif provider == "Google Gemini (Cloud AI)":
                    if not gemini_key:
                        st.error("Google API Key is required for Gemini provider.")
                    else:
                        topic, raw_resp = classify_with_gemini(news_text, gemini_key)
                        st.success(f"**Detected Topic:** {topic}")
                        with st.expander("Show Raw Model Response"):
                            st.write(raw_resp)

            except Exception as e:
                st.error(f"Error executing classifier: {e}")
                if provider == "Ollama (Local LLM)":
                    st.info("Tip: Verify Ollama is downloaded and running (`ollama serve`).")

# ---------- Footer ----------
st.markdown("---")
st.caption("News Topic Detector • Built with Python & Streamlit")