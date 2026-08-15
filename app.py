import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Fake News Generator & Detector",
    page_icon="📰",
    layout="centered",
)

DETECTOR_MODEL = "mrm8488/bert-tiny-finetuned-fake-news-detection"
GENERATOR_MODEL = "distilgpt2"


@st.cache_resource(show_spinner="Loading fake-news detection model...")
def load_detector():
    return pipeline("text-classification", model=DETECTOR_MODEL)


@st.cache_resource(show_spinner="Loading GPT-2 generator...")
def load_generator():
    return pipeline("text-generation", model=GENERATOR_MODEL)


def detect_news(text: str):
    classifier = load_detector()
    result = classifier(text, truncation=True, max_length=512)[0]
    label = result["label"].lower()
    confidence = result["score"]

    # This model uses LABEL_1 for real and LABEL_0 for fake.
    is_real = label in {"label_1", "real"}
    return is_real, confidence


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
                is_real, confidence = detect_news(text.strip())
                if is_real:
                    st.success(f"🟢 Likely REAL — confidence: {confidence:.1%}")
                else:
                    st.error(f"🔴 Likely FAKE — confidence: {confidence:.1%}")
                st.info(
                    "This is a machine-learning prediction, not a fact-check. "
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
