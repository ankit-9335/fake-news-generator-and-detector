import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Fake News Generator & Detector",
    page_icon="📰",
    layout="centered",
)

# The previous detector was too small and its LABEL_0/LABEL_1 mapping was
# not explicitly defined in the model config. That made our UI mapping unsafe.
# This model explicitly documents LABEL_0 = Fake and LABEL_1 = Real and reports
# 96.88% validation accuracy on its evaluation set.
DETECTOR_MODEL = "ungjus/Fake_News_BERT_Classifier"
GENERATOR_MODEL = "distilgpt2"


@st.cache_resource(show_spinner="Loading fake-news detection model...")
def load_detector():
    return pipeline(
        "text-classification",
        model=DETECTOR_MODEL,
        tokenizer=DETECTOR_MODEL,
    )


@st.cache_resource(show_spinner="Loading GPT-2 generator...")
def load_generator():
    return pipeline("text-generation", model=GENERATOR_MODEL)


def detect_news(text: str):
    classifier = load_detector()
    results = classifier(
        text,
        truncation=True,
        max_length=512,
        top_k=2,
    )

    # Transformers may return either a list of dictionaries or a nested list
    # depending on the installed pipeline version.
    if results and isinstance(results[0], list):
        results = results[0]

    scores = {item["label"].upper(): float(item["score"]) for item in results}

    fake_score = scores.get("LABEL_0", 0.0)
    real_score = scores.get("LABEL_1", 0.0)

    if fake_score >= real_score:
        return "FAKE", fake_score, real_score
    return "REAL", real_score, fake_score


def generate_news(prompt: str):
    generator = load_generator()
    result = generator(
        prompt,
        max_new_tokens=120,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        pad_token_id=generator.tokenizer.eos_token_id,
    )[0]["generated_text"]
    return result


st.title("📰 Fake News Generator & Detector")
st.caption(
    "Educational NLP demonstration — generated content should not be presented as real news."
)

# Keep the two functions separate so the app remains simple and easy to deploy.
tab_detect, tab_generate = st.tabs(["🔍 Detect News", "✍️ Generate News"])

with tab_detect:
    st.subheader("Detect whether a news article is likely fake or real")
    text = st.text_area(
        "News article or statement",
        height=220,
        placeholder="Paste the news content here...",
        key="detect_text",
    )

    if st.button("Check News", type="primary", key="check_news"):
        if not text.strip():
            st.warning("Please enter some news content first.")
        else:
            try:
                prediction, confidence, opposite_confidence = detect_news(text.strip())

                if prediction == "REAL":
                    st.success(f"🟢 Likely REAL — confidence: {confidence:.1%}")
                else:
                    st.error(f"🔴 Likely FAKE — confidence: {confidence:.1%}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Fake probability", f"{(confidence if prediction == 'FAKE' else opposite_confidence):.1%}")
                with col2:
                    st.metric("Real probability", f"{(confidence if prediction == 'REAL' else opposite_confidence):.1%}")

                st.info(
                    "This is a machine-learning prediction, not a fact-check. "
                    "The model learns patterns from its training data and can still be wrong. "
                    "Verify important claims using reliable sources."
                )
            except Exception as exc:
                st.error("The detector could not process this input.")
                st.exception(exc)

with tab_generate:
    st.subheader("Generate synthetic news text with GPT-2")
    prompt = st.text_area(
        "Headline or topic",
        height=140,
        placeholder="Example: Scientists discover a new species in the Pacific Ocean...",
        key="generate_prompt",
    )

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
st.caption(
    "Project: Fake News Generator and Detector | Python • Streamlit • Transformers • GPT-2"
)
