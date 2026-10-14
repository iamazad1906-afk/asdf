
from pathlib import Path
import sys

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_FILE = BASE_DIR / "spam_model.joblib"


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Dataset not found. Place SMSSpamCollection "
            "inside the data folder."
        )

    data = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    data = data.dropna()
    data["label"] = data["label"].str.strip().str.lower()
    data = data[data["label"].isin(["ham", "spam"])]

    return data


def train_model():
    data = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english"
            )
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Test Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved successfully!")


def predict_message():
    if not MODEL_FILE.exists():
        print("Train the model first using: python app.py train")
        return

    message = input("Enter a message: ").strip()

    if not message:
        print("Please enter a message.")
        return

    model = joblib.load(MODEL_FILE)
    prediction = model.predict([message])[0]

    if prediction == "spam":
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")


if __name__ == "__main__":
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "train"

    if command == "train":
        train_model()
    elif command == "predict":
        predict_message()
    else:
        print("Use: python app.py train")
        print("Or:  python app.py predict")
