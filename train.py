"""
Sentiment Analysis - Training Script
Dataset : IMDB Movie Reviews (50,000 reviews)
Models compared : Logistic Regression, Random Forest, SVM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import joblib
import re
import os

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_data(path: str = "data/IMDB_Dataset.csv") -> pd.DataFrame:
    """
    Loads the IMDB dataset from Kaggle.
    Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    """
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]} reviews")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    return df


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Cleans raw text."""
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning to the entire review column."""
    df = df.copy()
    df["review_clean"] = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    return df


# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────

def build_pipelines() -> dict:
    """Creates TF-IDF + classifier pipelines for each model."""
    tfidf_params = {
        "max_features": 20_000,
        "ngram_range": (1, 2),   # unigrams + bigrams
        "min_df": 3,
        "sublinear_tf": True,
    }

    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(C=1.0, random_state=42, max_iter=2000)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
    }


def evaluate(name: str, pipe, X_test, y_test) -> dict:
    """Evaluates a pipeline and prints metrics."""
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # AUC-ROC (if model supports predict_proba or decision_function)
    try:
        scores = pipe.decision_function(X_test)
        auc = roc_auc_score(y_test, scores)
    except AttributeError:
        scores = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, scores)

    print(f"AUC-ROC: {auc:.4f}")
    return {"name": name, "accuracy": acc, "auc": auc, "y_pred": y_pred}


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_confusion_matrix(name: str, y_test, y_pred, save_dir: str = "data"):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    fname = f"{save_dir}/confusion_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved: {fname}")


def plot_top_words(pipe, save_dir: str = "data", top_n: int = 20):
    """
    Displays the most important words from the Logistic Regression model.
    (Only compatible with LR)
    """
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]

    top_pos = np.argsort(coefs)[-top_n:]
    top_neg = np.argsort(coefs)[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, indices, title, color in zip(
        axes,
        [top_neg, top_pos],
        ["Words → Negative", "Words → Positive"],
        ["#e74c3c", "#2ecc71"]
    ):
        words = [feature_names[i] for i in indices]
        scores = [coefs[i] for i in indices]
        ax.barh(words, scores, color=color, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("TF-IDF Coefficient")

    plt.tight_layout()
    fname = f"{save_dir}/top_words.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

def main():
    # Load data
    df = load_data("data/IMDB_Dataset.csv")
    df = preprocess(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_clean"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Training and evaluation
    pipelines = build_pipelines()
    results = []

    for name, pipe in pipelines.items():
        print(f"\nTraining: {name}...")
        pipe.fit(X_train, y_train)
        result = evaluate(name, pipe, X_test, y_test)
        results.append(result)
        plot_confusion_matrix(name, y_test, result["y_pred"])

    # Best model → Logistic Regression (fastest + most interpretable)
    best_pipe = pipelines["Logistic Regression"]
    plot_top_words(best_pipe)

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipe, "models/sentiment_model.pkl")
    print("\nModel saved to models/sentiment_model.pkl")

    # Comparative summary
    print("\n" + "="*40)
    print("COMPARATIVE SUMMARY")
    print("="*40)
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        print(f"{r['name']:<25} Accuracy={r['accuracy']:.4f}  AUC={r['auc']:.4f}")


if __name__ == "__main__":
    main()
