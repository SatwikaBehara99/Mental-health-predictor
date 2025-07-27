import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# Load your dataset
df = pd.read_csv("mental-health.csv")  # Make sure the file exists and is clean
X = df["text"]
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Train model
X_vec = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vec, y_encoded)

# Save the model, vectorizer, and label encoder
joblib.dump(model, "model.pk1")
joblib.dump(vectorizer, "vectorizer.pk1")
joblib.dump(label_encoder, "label_encoder.pk1")

print("✅ Model, vectorizer, and label encoder saved successfully.")
