import os
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="NewsMorph — Fake News Detector", page_icon="📰", layout="centered")

# Set NEWSMORPH_MODEL_ID in Streamlit secrets/environment after fine-tuning.
# Example: ankit-9335/newsmorph-fake-news-detector
MODEL_ID = os.getenv("NEWSMORPH_MODEL_ID", "ankit-9335/newsmorph-fake-news-detector")
GENERATOR_MODEL = "distilgpt2"

@st.cache_resource(show_spinner="Loading fine-tuned fake-news detector...")
def load_detector():
    return pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID, top_k=None)

@st.cache_resource(show_spinner="Loading text generator...")
def load_generator():
    return pipeline("text-generation", model=GENERATOR_MODEL)

def detect_news(text: str):
    results = load_detector()(text, truncation=True, max_length=512)
    if results and isinstance(results[0], list):
        results = results[0]
    scores = {item["label"].upper(): float(item["score"]) for item in results}
    fake = scores.get("FAKE", 0.0)
    real = scores.get("REAL", 0.0)
    margin = abs(fake - real)
    prediction = "UNCERTAIN" if margin < 0.20 else ("FAKE" if fake > real else "REAL")
    return prediction, fake, real

def generate_news(prompt: str):
    generator = load_generator()
    return generator(prompt, max_new_tokens=120, num_return_sequences=1, do_sample=True,
                      temperature=0.8, top_k=50, top_p=0.95, no_repeat_ngram_size=2,
                      pad_token_id=generator.tokenizer.eos_token_id)[0]["generated_text"]

st.title("📰 NewsMorph")
st.caption("AI-powered fake-news classification. A model prediction is not the same as factual verification.")
tab_detect, tab_generate = st.tabs(["🔍 Detect News", "✍️ Generate News"])

with tab_detect:
    st.subheader("Classify a news article")
    text = st.text_area("News article or statement", height=250, placeholder="Paste a news article here...", key="detect_text")
    if st.button("Check News", type="primary", key="check_news"):
        if not text.strip():
            st.warning("Please enter some news content first.")
        else:
            try:
                prediction, fake, real = detect_news(text.strip())
                c1, c2 = st.columns(2)
                with c1: st.metric("Fake probability", f"{fake:.1%}")
                with c2: st.metric("Real probability", f"{real:.1%}")
                if prediction == "FAKE":
                    st.error("🔴 Model classification: FAKE")
                elif prediction == "REAL":
                    st.success("🟢 Model classification: REAL")
                else:
                    st.warning("🟡 Model classification: UNCERTAIN")
                st.info("This classifier learns patterns from labeled news data. It does not independently verify whether an event actually happened.")
            except Exception as exc:
                st.error("The detector could not process this input.")
                st.exception(exc)

with tab_generate:
    st.subheader("Generate synthetic news text")
    prompt = st.text_area("Headline or topic", height=140, placeholder="Example: Scientists discover a new species in the Pacific Ocean...", key="generate_prompt")
    if st.button("Generate", type="primary", key="generate_news"):
        if not prompt.strip():
            st.warning("Please enter a headline or topic first.")
        else:
            try:
                with st.spinner("Generating synthetic text..."):
                    generated = generate_news(prompt.strip())
                st.text_area("Generated content", generated, height=280)
            except Exception as exc:
                st.error("The generator could not produce text.")
                st.exception(exc)

st.divider()
st.caption("NewsMorph | Python • Streamlit • Transformers • DistilBERT • GPT-2")
