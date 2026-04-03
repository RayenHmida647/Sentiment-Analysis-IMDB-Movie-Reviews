"""
app.py — Streamlit interface for sentiment analysis
Run with: streamlit run app.py
"""

import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .result-pos {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #155724;
    }
    .result-neg {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        color: #721c24;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Loads the trained model (or returns None if not found)."""
    try:
        model = joblib.load("models/sentiment_model.pkl")
        return model, True
    except FileNotFoundError:
        return None, False


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(model, text: str):
    """Returns (label, positive_proba)."""
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]

    try:
        proba = model.predict_proba([cleaned])[0][1]
    except AttributeError:
        score = model.decision_function([cleaned])[0]
        proba = 1 / (1 + np.exp(-score))  # sigmoid

    label = "Positive ✅" if pred == 1 else "Negative ❌"
    return label, float(proba)


# ─────────────────────────────────────────────
# PRE-FILLED EXAMPLES
# ─────────────────────────────────────────────

EXAMPLES = {
    "Positive review (film)": "This movie was absolutely breathtaking! The performances were outstanding and the cinematography was stunning. I couldn't take my eyes off the screen for a single moment.",
    "Negative review (film)": "What a waste of time. The plot made no sense, the acting was terrible and the special effects looked cheap. I almost fell asleep halfway through.",
    "Mixed review": "The film had some interesting ideas but the execution was mediocre. Some scenes were good, others dragged on too long. A mixed bag overall.",
    "Positive tweet": "Just saw the new Marvel movie and I'm absolutely blown away! Best superhero film in years! #Marvel #Cinema",
    "Negative tweet": "Worst movie of the year. Total disappointment. Avoid at all costs.",
}

# ─────────────────────────────────────────────
# MAIN INTERFACE
# ─────────────────────────────────────────────

model, model_loaded = load_model()

st.markdown('<p class="main-title">🎬 Sentiment Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Model: TF-IDF + Logistic Regression · Dataset: IMDB 50K reviews</p>', unsafe_allow_html=True)

# Warning if model not found
if not model_loaded:
    st.warning(
        "⚠️ Model not found. Run `python src/train.py` first to train the model."
    )

# ─── Tabs ───
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Batch Analysis", "📖 About the Project"])

# ─────────────────────────────────────────────
# TAB 1: SINGLE PREDICTION
# ─────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Enter your text")

        example_choice = st.selectbox(
            "Load an example:", ["(free text)"] + list(EXAMPLES.keys())
        )
        default_text = EXAMPLES.get(example_choice, "")

        user_text = st.text_area(
            "Review or tweet to analyze:",
            value=default_text,
            height=150,
            placeholder="Enter your text in English here...",
        )

        analyze_btn = st.button("🔎 Analyze", type="primary", disabled=not model_loaded)

    with col2:
        st.subheader("Result")

        if analyze_btn and user_text.strip():
            label, proba = predict(model, user_text)
            is_positive = proba >= 0.5
            confidence = proba if is_positive else 1 - proba

            css_class = "result-pos" if is_positive else "result-neg"
            st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)

            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(proba)
            st.caption(f"Positive score: {proba:.3f}")

            # Cleaned text
            with st.expander("View preprocessed text"):
                st.code(clean_text(user_text))

        elif analyze_btn:
            st.warning("Please enter some text.")
        else:
            st.info("Enter a text and click Analyze.")

# ─────────────────────────────────────────────
# TAB 2: BATCH ANALYSIS (CSV)
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Analyze a CSV file")
    st.info("The CSV file must contain a `text` column with English reviews.")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded and model_loaded:
        df_upload = pd.read_csv(uploaded)

        if "text" not in df_upload.columns:
            st.error("Column `text` not found in the file.")
        else:
            with st.spinner("Analyzing..."):
                texts = df_upload["text"].fillna("").astype(str).tolist()
                cleaned = [clean_text(t) for t in texts]
                preds = model.predict(cleaned)

                try:
                    probas = model.predict_proba(cleaned)[:, 1]
                except AttributeError:
                    scores = model.decision_function(cleaned)
                    probas = 1 / (1 + np.exp(-scores))

                df_upload["predicted_sentiment"] = ["Positive" if p == 1 else "Negative" for p in preds]
                df_upload["positive_score"] = probas.round(3)

            # Stats
            col1, col2, col3 = st.columns(3)
            n_pos = (preds == 1).sum()
            n_neg = (preds == 0).sum()
            col1.metric("Total analyzed", len(preds))
            col2.metric("✅ Positive", f"{n_pos} ({n_pos/len(preds):.0%})")
            col3.metric("❌ Negative", f"{n_neg} ({n_neg/len(preds):.0%})")

            # Chart
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["Positive", "Negative"], [n_pos, n_neg], color=["#28a745", "#dc3545"])
            ax.set_ylabel("Number of reviews")
            ax.set_title("Predicted sentiment distribution")
            st.pyplot(fig)

            # Table
            st.dataframe(df_upload[["text", "predicted_sentiment", "positive_score"]].head(20))

            # Export
            csv_out = df_upload.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results",
                csv_out,
                "sentiment_results.csv",
                "text/csv",
            )

# ─────────────────────────────────────────────
# TAB 3: ABOUT
# ─────────────────────────────────────────────
with tab3:
    st.subheader("Project architecture")

    st.markdown("""
    ### Dataset
    - **IMDB Movie Reviews** — 50,000 movie reviews (25k positive / 25k negative)
    - Source: [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

    ### Processing pipeline
    1. **Cleaning**: HTML removal, punctuation stripping, lowercasing
    2. **Vectorization**: TF-IDF (20,000 features, unigrams + bigrams)
    3. **Model**: Logistic Regression (C=1.0)

    ### Performance (test set — 10,000 reviews)
    | Metric | Score |
    |---|---|
    | Accuracy | ~89% |
    | Precision (positive) | ~89% |
    | Recall (positive) | ~90% |
    | AUC-ROC | ~96% |

    ### Models compared
    - Logistic Regression ✅ (best precision/speed trade-off)
    - Linear SVM
    - Random Forest

    ### Project structure
    ```
    sentiment_analysis/
    ├── app.py                  # Streamlit interface
    ├── src/
    │   └── train.py            # Training + evaluation
    ├── notebooks/
    │   └── exploration.ipynb   # EDA + experiments
    ├── data/
    │   └── IMDB_Dataset.csv    # Dataset (download from Kaggle)
    ├── models/
    │   └── sentiment_model.pkl # Serialized model
    ├── requirements.txt
    └── README.md
    ```

    ### Possible extensions (next steps)
    - 🤗 Use BERT via HuggingFace (`distilbert-base-uncased-finetuned-sst-2-english`)
    - 🌍 French language support with CamemBERT
    - 📡 Connect Twitter/X API for real-time data
    - 🐳 Dockerize the application
    """)
