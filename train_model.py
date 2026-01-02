import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv("data.csv")

X = data["text"]
y = data["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 🔥 VERY IMPORTANT FIX
)

# Improved pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"  # 🔥 MAJOR FIX
    ))
])

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "sentiment_model.pkl")

print("✅ Model trained successfully!")

