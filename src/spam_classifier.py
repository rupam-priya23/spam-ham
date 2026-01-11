import os
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords (runs once, safe to keep)
nltk.download("stopwords")


def clean_text(text: str) -> str:
    """
    Clean input text by:
    - Lowercasing
    - Removing special characters
    - Removing extra spaces
    """
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def train_model():
    """
    Loads data, trains the spam classifier model,
    and returns model, vectorizer, and clean_text function.
    """

    # Get absolute path to project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")

    # Load dataset
    data = pd.read_csv(DATA_PATH, encoding="latin-1")

    # Keep only required columns
    data = data[["v1", "v2"]]
    data.columns = ["label", "message"]

    # Encode labels
    data["label"] = data["label"].map({"ham": 0, "spam": 1})

    # Clean messages
    data["message"] = data["message"].apply(clean_text)

    # Convert text to numbers using TF-IDF
    vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
    X = vectorizer.fit_transform(data["message"])
    y = data["label"]

    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X, y)

    return model, vectorizer, clean_text


# Allow running this file directly (optional test)
if __name__ == "__main__":
    model, vectorizer, clean_fn = train_model()

    test_messages = [
        "Congratulations! You won a free iPhone",
        "Hey, are we meeting today?",
        "WIN CASH NOW!!! Text WIN to 99999"
    ]

    for msg in test_messages:
        cleaned = clean_fn(msg)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        result = "SPAM" if prediction == 1 else "NOT SPAM"
        print(f"Message: {msg}")
        print(f"Prediction: {result}\n")
