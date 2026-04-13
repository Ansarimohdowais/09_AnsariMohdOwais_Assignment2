
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "asha_bhosle_tweets.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "tweet" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'tweet' and 'sentiment' columns.")
    df = df.dropna(subset=["tweet", "sentiment"]).copy()
    df["clean_tweet"] = df["tweet"].apply(clean_text)
    return df


def plot_label_distribution(df: pd.DataFrame) -> None:
    counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
    plt.figure(figsize=(7, 4))
    counts.plot(kind="bar")
    plt.title("Label Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "label_distribution.png", dpi=200)
    plt.close()


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("clf", model)
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0
    )
    report = classification_report(y_test, pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, pred, labels=["negative", "neutral", "positive"])
    return {
        "name": name,
        "pipeline": pipeline,
        "predictions": pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = ["negative", "neutral", "positive"]
    plt.xticks(range(3), ticks, rotation=20)
    plt.yticks(range(3), ticks)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i][j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=200)
    plt.close()


def save_model_comparison(results):
    names = [r["name"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    x = range(len(names))
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - width/2 for i in x], precisions, width=width, label="Precision")
    plt.bar([i + width/2 for i in x], recalls, width=width, label="Recall")
    plt.xticks(list(x), names, rotation=15)
    plt.ylim(0, 1.05)
    plt.title("Model Comparison on Test Set")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=200)
    plt.close()


def main():
    df = load_data()
    plot_label_distribution(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_tweet"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )

    models = [
        ("Naive Bayes", MultinomialNB()),
        ("SVM", LinearSVC()),
        ("Logistic Regression", LogisticRegression(max_iter=2000)),
    ]

    results = []
    for name, model in models:
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)
        save_confusion_matrix(
            result["confusion_matrix"],
            f"{name} Confusion Matrix",
            f"confusion_{name.lower().replace(' ', '_')}.png",
        )

    save_model_comparison(results)

    summary = {
        r["name"]: {
            "precision_macro": round(float(r["precision"]), 4),
            "recall_macro": round(float(r["recall"]), 4),
            "f1_macro": round(float(r["f1"]), 4),
        }
        for r in results
    }

    out = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": summary,
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    metrics_df = pd.DataFrame([
        {
            "Model": r["name"],
            "Precision (macro)": round(float(r["precision"]), 4),
            "Recall (macro)": round(float(r["recall"]), 4),
            "F1 (macro)": round(float(r["f1"]), 4),
        }
        for r in results
    ])
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print(metrics_df.to_string(index=False))
    print("\nSaved outputs in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
