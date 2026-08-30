import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import streamlit as st
from preprocessing import clean_text

@st.cache_resource
def train_model(data_path):
    """
    Trains the TF-IDF and Logistic Regression model on the provided dataset.
    Cached so it doesn't retrain on every Streamlit interaction.
    """
    if not os.path.exists(data_path):
        return None, None, None, {"error": "Dataset not found."}
        
    try:
        df = pd.read_csv(data_path)
        
        # Ensure required columns exist
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            return None, None, None, {"error": "Dataset must contain 'text' and 'sentiment' columns."}
            
        # Drop rows with missing values in text or sentiment
        df = df.dropna(subset=['text', 'sentiment'])
        
        # Preprocess all texts
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        X = df['cleaned_text']
        y = df['sentiment']
        
        # Train/Test Split for Evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # TF-IDF Feature Extraction
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Logistic Regression Model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_tfidf, y_train)
        
        # Model Evaluation
        y_pred = model.predict(X_test_tfidf)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=model.classes_),
            "classes": model.classes_,
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
        
        return vectorizer, model, df, metrics
    except Exception as e:
        return None, None, None, {"error": str(e)}

def predict_sentiment(text, vectorizer, model):
    """
    Predicts sentiment for a given text using the trained model.
    """
    if not text or not vectorizer or not model:
        return None, 0.0
        
    cleaned_text = clean_text(text)
    
    # If text is empty after cleaning
    if not cleaned_text:
         return "Neutral", 0.0 # Default fallback
         
    features = vectorizer.transform([cleaned_text])
    
    prediction = model.predict(features)[0]
    
    # Get probability/confidence
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    return prediction, confidence
