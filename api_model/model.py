import pandas as pd
import os
import pickle

import re
import string
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score


DATA_PATH = "data"
MODEL_PATH = "models"


def load_data():
    dataframes = []
    genres = ["bossa_nova", "funk", "gospel", "sertanejo"]
    for genre in genres:
        df = pd.read_csv(os.path.join(DATA_PATH, f"{genre}.csv"))
        df["genre"] = genre
        dataframes.append(df)
    df_final = pd.concat(dataframes, ignore_index=True)
    return df_final

nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    return " ".join(tokens)

def preprocess_data(df):
    df.dropna(subset=["lyric"], inplace=True)
    df["lyric"] = df["lyric"].apply(preprocess_text)
    return df


def train_and_evaluate(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ])

    param_grid = [
        {
            "clf": [MultinomialNB()],
            'clf__alpha': [0.01, 0.1, 1, 10]
        },
        {
            "clf": [LogisticRegression(max_iter=1000)],
            'clf__C': [0.1, 1, 10],
            'clf__max_iter': [100, 500, 1000]
        },
        {
            "clf": [RandomForestClassifier()],
            'clf__n_estimators': [100, 200,500],
            'clf__max_depth': [5, 10, 20],
            'clf__min_samples_split': [2, 5, 10]
        },
        {
            "clf": [LinearSVC()],
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__max_iter': [1000, 2000, 5000],
            'clf__loss': ['hinge', 'squared_hinge']
        },
    ]

    grid = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5), scoring="accuracy", verbose=1, n_jobs=1)
    grid.fit(X_train, y_train)

    print("Melhor modelo:", grid.best_estimator_)
    y_pred = grid.predict(X_test)
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

    return grid.best_estimator_

def save_model(model):
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    with open(os.path.join(MODEL_PATH, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def main():
    df = load_data()
    df = preprocess_data(df)

    X = df["lyric"]
    y = df["genre"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    best_model = train_and_evaluate(X_train, X_test, y_train, y_test)
    save_model(best_model)

if __name__ == "__main__":
    main()
