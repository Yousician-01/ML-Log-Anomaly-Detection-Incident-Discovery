import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Configs

INPUT_PATH = "data/processed/unified_logs.parquet"
OUTPUT_DIR = Path("data/processed/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Data

df = pd.read_parquet(INPUT_PATH)
texts = df['template'].astype(str).tolist()

# TF-IDF Vectorization

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    max_features=5000
    )

X = vectorizer.fit_transform(texts)

# Save Embeddings and Vectorizer

joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.joblib")
joblib.dump(X, OUTPUT_DIR / "tfidf_embeddings.joblib")

print("TF-IDF embeddings and vectorizer saved successfully.")
print(f"Embeddings shape: {X.shape}")