print("Script started")

import os
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download("stopwords")

# ---------- PATH FIX ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")

print("Looking for file at:", DATA_PATH)

# ---------- LOAD DATA ----------
data = pd.read_csv(DATA_PATH, encoding="latin-1")

data = data[["v1", "v2"]]
data.columns = ["label", "message"]

# ---------- LABEL ENCODING ----------
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

data["message"] = data["message"].apply(clean_text)

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X = vectorizer.fit_transform(data["message"])
y = data["label"]

# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- MODEL ----------
model = MultinomialNB()
model.fit(X_train, y_train)

# ---------- ACCURACY ----------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------- TEST ----------
def predict_message(msg):
    msg = clean_text(msg)
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)
    return "Spam" if prediction[0] == 1 else "Not Spam"

print(predict_message("Congratulations! You won a free iPhone"))
print(predict_message("Hey, are we meeting today?"))
