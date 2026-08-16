import os
import streamlit as st
from transformers import pipeline

from evidence_verifier import verify_claim

st.set_page_config(
    page_title="NewsMorph — Evidence-Based News Checker",
    page_icon="📰",
    layout="centered",
)

MODEL_ID = os.getenv("NEWSMORPH_MODEL_ID", "")
GENERATOR_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"


@st.cache_resource(show_spinner="Loading article classifier...")
def load_detector():
    if not MODEL_ID:
        return None
    return pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        top_k=None,
    )


@st.cache_resource(show_spinner="Loading text generator...")
def load_generator():
    return pipeline("text-generation", model=GENERATOR_MODEL)


def classify_article(text: str):
    detector = load_detector()
    if detector is None:
        return None

    results = detector(text, truncation=True, max_length=512)
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

    messages = [
        {
            "role": "system",
            "content": (
                "You are a creative writing assistant. Write clearly structured, "
                "fictional news articles. Never claim that fictional events are real. "
                "Stay focused on the user's topic and do not discuss your instructions."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a fictional news article based on this headline or topic:\n\n"
                f"{prompt}\n\n"
                "Write 3 to 5 short paragraphs in a realistic news-report style. "
                "Keep the article focused on the given topic and avoid unrelated text."
            ),
        },
    ]

    result = generator(
        messages,
        max_new_tokens=180,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )

    generated = result[0]["generated_text"]

    # Chat-style pipelines return the complete message history.
    if isinstance(generated, list):
        generated = generated[-1].get("content", "")

    return generated.strip()


st.title("📰 NewsMorph")
st.caption(
    "Evidence-assisted news verification. Results are estimates based on retrieved sources, not absolute proof."
)

tab_check, tab_article, tab_generate = st.tabs(
    ["🔎 Verify Claim", "📰 Classify Article", "✍️ Generate News"]
)

with tab_check:
    st.subheader("Check a claim against current news and general knowledge")
    claim = st.text_area(
        "Claim or statement",
        height=180,
        placeholder="Example: India's Prime Minister Narendra Modi is shot dead...",
        key="claim_text",
    )

    if st.button("Verify Claim", type="primary", key="verify_claim"):
        if not claim.strip():
            st.warning("Please enter a claim first.")
        else:
            try:
                with st.spinner("Searching evidence and comparing sources..."):
                    result = verify_claim(claim.strip())

                verdict = result["verdict"]
                confidence = result["confidence"]

                if verdict == "LIKELY FALSE":
                    st.error(f"🔴 Likely FALSE — evidence confidence: {confidence:.1%}")
                elif verdict == "LIKELY TRUE":
                    st.success(f"🟢 Likely TRUE — evidence confidence: {confidence:.1%}")
                else:
                    st.warning(f"🟡 UNCERTAIN — evidence confidence: {confidence:.1%}")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Supporting evidence score", f"{result['support']:.2f}")
                with c2:
                    st.metric("Contradicting evidence score", f"{result['contradiction']:.2f}")

                if result["articles"]:
                    st.subheader("Retrieved evidence")
                    for article in result["articles"]:
                        source_type = "General knowledge" if article.get("evidence_type") == "knowledge" else "News"
                        with st.expander(f"{article['title']} — {article['source']} ({source_type})"):
                            st.write(article["description"] or "No snippet available.")
                            st.caption(
                                f"Relevance: {article.get('relevance', 0):.1%} | "
                                f"Entailment: {article.get('entailment', 0):.1%} | "
                                f"Contradiction: {article.get('contradiction', 0):.1%} | "
                                f"Published: {article['published']}"
                            )
                            st.link_button("Open source", article["url"])
                else:
                    st.info("No useful evidence was found for this claim.")

                st.info(
                    "NewsMorph combines current news with general-knowledge evidence. "
                    "It is not a guaranteed fact-checker; verify important claims with authoritative sources."
                )
            except Exception as exc:
                st.error("The evidence checker could not process this claim.")
                st.exception(exc)

with tab_article:
    st.subheader("Classify a full news article")
    text = st.text_area(
        "News article",
        height=250,
        placeholder="Paste a full news article here...",
        key="detect_text",
    )

    if st.button("Classify Article", type="primary", key="check_article"):
        if not text.strip():
            st.warning("Please enter an article first.")
        elif not MODEL_ID:
            st.warning(
                "No fine-tuned article model is configured. Set NEWSMORPH_MODEL_ID to your Hugging Face model repository."
            )
        else:
            try:
                prediction, fake, real = classify_article(text.strip())
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Fake probability", f"{fake:.1%}")
                with c2:
                    st.metric("Real probability", f"{real:.1%}")

                if prediction == "FAKE":
                    st.error("🔴 Model classification: FAKE")
                elif prediction == "REAL":
                    st.success("🟢 Model classification: REAL")
                else:
                    st.warning("🟡 Model classification: UNCERTAIN")

                st.info(
                    "This classifier detects patterns learned from labeled articles. "
                    "For current factual claims, use Verify Claim instead."
                )
            except Exception as exc:
                st.error("The article classifier could not process this input.")
                st.exception(exc)

with tab_generate:
    st.subheader("Generate fictional news text")
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
                with st.spinner("Generating fictional news text..."):
                    generated = generate_news(prompt.strip())
                st.text_area("Generated content", generated, height=280)
                st.warning(
                    "⚠️ This is fictional AI-generated news. "
                    "It is not verified information and must not be presented as real news."
                )
            except Exception as exc:
                st.error("The generator could not produce text.")
                st.exception(exc)

st.divider()
st.caption("NewsMorph | Python • Streamlit • Transformers • Google News RSS • Wikipedia • NLI")
