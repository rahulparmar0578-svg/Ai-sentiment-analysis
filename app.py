"""
app.py — Streamlit UI for AI-Based Sentiment Analysis System.
Loads the saved Keras model + tokenizer and runs inference.
"""

import json
import os
import pickle
import re

import streamlit as st

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics.json")

MAX_LEN = 200  # must match train.py

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Based Sentiment Analysis System",
    page_icon="🎬",
    layout="centered",
)

# ── Minimal CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"], button, input, textarea {
    font-family: "Segoe UI", Inter, Arial, sans-serif !important;
  }
  body { line-height: 1.55; }
  .block-container { max-width: 760px; padding-top: 2rem; }
  h1 { color: #1a56db; font-size: 1.9rem; font-weight: 750; letter-spacing: -0.035em; }
  .result-positive {
    background: #e6f4ea; border: 1px solid #a8d5b5; border-radius: 8px;
    padding: 16px 20px; text-align: center; margin-top: 12px; color: #174a24;
  }
  .result-negative {
    background: #fce8e6; border: 1px solid #f5a6a0; border-radius: 8px;
    padding: 16px 20px; text-align: center; margin-top: 12px; color: #6b211d;
  }
  .result-positive span { color: #1e7a34; font-size: 1.4rem; font-weight: 700; }
  .result-negative span { color: #c5221f; font-size: 1.4rem; font-weight: 700; }
  .about-box {
    background: #f1f5ff; border-radius: 8px; padding: 16px 20px; margin-top: 18px;
    font-size: 0.92rem; color: #333;
  }
</style>
""", unsafe_allow_html=True)

# ── Helpers ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    import tensorflow as tf  # noqa: lazy import so missing-model error is clear
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: keep ref

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        return None, None, None

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    return model, tokenizer, metrics


def predict(text, model, tokenizer):
    """Run the same preprocessing as training, then inference."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    score = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if score >= 0.5 else "Negative"
    return label, score


# ── Header ──────────────────────────────────────────────────────────
st.title("AI-Based Sentiment Analysis System")
st.caption(
    "Enter a movie review below and the trained LSTM model will classify "
    "its sentiment as **Positive** or **Negative**."
)
st.divider()

# ── Check artifacts exist ───────────────────────────────────────────
model, tokenizer, metrics = load_artifacts()

if model is None or tokenizer is None:
    st.warning(
        "⚠️ Trained model not found. Please train the model first:\n\n"
        "```powershell\n"
        "python train.py\n"
        "```\n\n"
        "This downloads the IMDb dataset, trains the LSTM model, and saves "
        "artifacts to `model/` and `results/`."
    )
    st.stop()

# ── Input ───────────────────────────────────────────────────────────
sample_col1, sample_col2 = st.columns(2)
if sample_col1.button("Try a positive example", use_container_width=True):
    st.session_state.review_text = (
        "I absolutely loved this movie. The performances were outstanding, "
        "the story was captivating, and I would happily watch it again."
    )
if sample_col2.button("Try a negative example", use_container_width=True):
    st.session_state.review_text = (
        "This movie was awful. I hated every minute; the acting was terrible "
        "and the story was boring."
    )

review = st.text_area(
    "Movie review",
    placeholder="e.g.  This movie was absolutely fantastic! Great acting and a wonderful story.",
    height=140,
    key="review_text",
)

col_btn, _ = st.columns([1, 2])
with col_btn:
    analyze = st.button("Analyze Sentiment", type="primary", use_container_width=True)

# ── Inference ───────────────────────────────────────────────────────
if analyze:
    if not review or not review.strip():
        st.error("Please enter a review before analyzing.")
    else:
        label, score = predict(review, model, tokenizer)
        confidence = score if label == "Positive" else 1 - score

        if label == "Positive":
            st.markdown(
                f'<div class="result-positive"><span>😊 Positive</span><br>'
                f'Model confidence: {confidence:.1%}<br>'
                f'<small>Raw output score: {score:.4f}</small></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-negative"><span>😞 Negative</span><br>'
                f'Model confidence: {confidence:.1%}<br>'
                f'<small>Raw output score: {score:.4f}</small></div>',
                unsafe_allow_html=True,
            )
        st.caption("Confidence reflects the model's output score, not a guarantee of correctness.")
        st.progress(confidence, text=f"Prediction confidence: {confidence:.1%}")
        if confidence < 0.6:
            st.warning("Low confidence: this review is close to the model's decision boundary. Try a longer, more specific review.")

# ── Metrics ─────────────────────────────────────────────────────────
if metrics:
    st.subheader("Model Evaluation (held-out test set)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("F1-score", f"{metrics['f1_score']:.2%}")

    cm = metrics["confusion_matrix"]
    # Simple text-based confusion matrix — no matplotlib needed at runtime
    st.markdown(
        f"""
|  | Pred Negative | Pred Positive |
|---|---|---|
| **Actual Negative** | {cm[0][0]} | {cm[0][1]} |
| **Actual Positive** | {cm[1][0]} | {cm[1][1]} |
"""
    )
    st.caption("Rows = true label, Columns = predicted label.")

# ── About ───────────────────────────────────────────────────────────
st.markdown("""
<div class="about-box">
<strong>About this project</strong><br><br>
<strong>NLP pipeline:</strong> raw text → lowercasing & cleaning → Keras Tokenizer
(word → integer IDs) → padding/truncation to 200 tokens.<br>
<strong>Deep learning model:</strong> Embedding (10k × 128) → LSTM (64 units) →
Dropout (0.5) → Dense sigmoid. Trained on the IMDb 50k movie-review dataset
(25k train / 25k test) with early stopping.<br>
<strong>Dataset:</strong> <em>IMDb Movie Reviews</em> — 50,000 labeled reviews
via <code>tf.keras.datasets.imdb</code>.
</div>
""", unsafe_allow_html=True)
