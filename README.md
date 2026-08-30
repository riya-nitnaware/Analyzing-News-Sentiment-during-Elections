# Analyzing News Sentiment During Elections

An academic, production-grade Natural Language Processing (NLP) web application designed to analyze, classify, and visualize the sentiment of election-related news articles into **Positive**, **Neutral**, or **Negative** categories. Built with Python, Streamlit, Scikit-learn, and NLTK.

---

## 📌 Project Overview

During election periods, media coverage significantly influences public discourse. Understanding the overall tone and sentiment expressed in election news articles provides valuable analytical insights for researchers, students, and analysts.

This application provides a complete end-to-end NLP solution that allows users to:
1. Input news headlines, full articles, or both.
2. Process text through an automated NLP cleaning and TF-IDF feature extraction pipeline.
3. Predict sentiment using a trained **Logistic Regression** machine learning model.
4. View model confidence and class probability distributions.
5. Save personal analysis records to a private user history.
6. Generate and download professionally formatted **PDF reports** for academic presentations or documentation.
7. Compare multiple past analyses side-by-side.

> **Important Note:** This system analyzes textual sentiment expressed within news articles. It does **NOT** predict election outcomes, voter behavior, political fairness, or factual accuracy.

---

## ✨ Features

- **🔒 Authentication & User Profiles:** Built-in sign-up and login authentication system with hashed credentials.
- **📰 Headline & Article Sentiment Analysis:** Flexible input accepting headlines only, full news text, or both.
- **📊 Real Confidence & Probability Distribution:** Real-time classification with progress bars for Positive, Neutral, and Negative probabilities.
- **💾 Personal Analysis Persistence:** Save and manage personal analysis records securely without exposing raw dataset rows.
- **📥 Downloadable PDF Reports:** Generate clean, academic PDF reports containing text stats, sentiment results, probabilities, and disclaimers.
- **🔄 Side-by-Side Comparison:** Compare any two previously saved news analyses across metrics like sentiment, confidence, word count, and date.
- **📈 Personal Analytics Dashboard:** Visual insights (donut distribution, sentiment over time) based strictly on user's saved analysis history.
- **ℹ️ Academic Documentation Page:** Comprehensive overview of the NLP pipeline, technology stack, training dataset, and project scope.

---

## 🔄 How It Works & NLP Pipeline

The application processes news articles through a sequential 7-step Natural Language Processing and Machine Learning pipeline:

```
┌─────────────────────────────────────────────────────────┐
│               1. Input Headline / Article               │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                 2. Text Preprocessing                   │
│   • Lowercased text normalization                        │
│   • URL, HTML tag, and special character removal       │
│   • Tokenization via NLTK word_tokenize                 │
│   • Stopword removal (English NLTK corpus)              │
│   • WordNet Lemmatization                               │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│             3. TF-IDF Feature Extraction                │
│   • Unigrams & Bigrams (ngram_range=(1, 2))             │
│   • Sublinear TF scaling (sublinear_tf=True)            │
│   • Document Frequency filtering (min_df=2, max_df=0.9) │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│           4. Machine Learning Classification            │
│   • Logistic Regression Classifier                      │
│   • Multi-class Probability Estimation (predict_proba)  │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│             5. Sentiment & Confidence Result            │
│   • Class prediction: Positive / Neutral / Negative     │
│   • Max confidence score + full class probabilities     │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│               6. Personal History & Storage             │
│   • Saved to user-specific JSON storage                 │
└────────────────────────────┬────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│          7. PDF Report Generation & Comparison          │
│   • Clean academic PDF report via fpdf2                 │
│   • Multi-analysis comparison engine                    │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Programming Language** | Python 3 |
| **Web Framework** | Streamlit |
| **Natural Language Processing** | NLTK (Tokenization, Stopwords, Lemmatization) |
| **Machine Learning** | Scikit-learn (Logistic Regression, TF-IDF Vectorizer) |
| **Data Processing** | Pandas, NumPy |
| **Visualizations** | Plotly Express, WordCloud, Matplotlib |
| **PDF Generation** | fpdf2 (with project-local TTF fonts) |
| **Authentication & Storage** | Local JSON store (`users.json`, `history.json`), SHA-256 password hashing |

---

## 📁 Project Structure

```
election-news-sentiment-analysis/
│
├── .streamlit/
│   └── config.toml           # Theme configuration (Navy/Blue palette)
│
├── assets/
│   ├── fonts/
│   │   ├── arial.ttf         # Regular TTF font for PDF Unicode support
│   │   ├── arialbd.ttf       # Bold TTF font
│   │   └── ariali.ttf        # Italic TTF font
│   └── style.css             # Custom CSS styling for Streamlit elements
│
├── data/
│   ├── news_data.csv         # Training dataset (110 election news articles)
│   ├── news_data_template.csv# Template for custom dataset format
│   ├── users.json            # User account store
│   └── history.json          # Personal analysis history store
│
├── app.py                    # Main Streamlit application entry point & router
├── auth.py                   # User authentication & registration module
├── preprocessing.py          # NLTK text cleaning and normalization pipeline
├── nlp_model.py              # ML model training, caching, & prediction engine
├── evaluation.py             # Model performance metrics & confusion matrix logic
├── visualizations.py        # Plotly charts and WordCloud generation
├── history_manager.py        # History CRUD operations per user
├── pdf_generator.py          # Robust Unicode PDF report generator
├── requirements.txt          # Python dependency specifications
├── .gitignore                # Git exclusion rules
└── README.md                 # Project documentation
```

---

## 💻 Installation

### Prerequisites
- **Python 3.9+** installed on your system.
- **Git** installed for repository cloning.

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd election-news-sentiment-analysis
```

### Step 2: Create a Virtual Environment

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Launch the Streamlit web application:

```bash
streamlit run app.py
```

After running the command, open your browser and navigate to:
```
http://localhost:8501
```

---

## 📖 How to Use

1. **Sign In / Register:**
   - Use the default credentials or register a new account on the Sign Up page.
   - **Default Admin Username:** `admin`
   - **Default Admin Password:** `admin123`

2. **Analyze Election News:**
   - Navigate to **📰 Analyze News** in the sidebar.
   - Enter a news headline, paste full article text, or both.
   - Click **Analyze Sentiment →**.

3. **Review Results:**
   - View the predicted sentiment (**POSITIVE**, **NEUTRAL**, or **NEGATIVE**).
   - Inspect the confidence percentage and probability breakdown bars.

4. **Save & Export:**
   - Click **💾 Save Analysis** to store the result in your personal history.
   - Click **📥 Download Report** to generate an academic PDF report.

5. **Explore Personal Dashboard & Compare:**
   - Visit **📊 Analysis Dashboard** to view personal analytics.
   - Visit **🔄 Compare Analyses** to evaluate two past articles side by side.

---

## 📊 Dataset

The model is trained on `data/news_data.csv`, a curated dataset of **110 election-related news articles**:

- **Positive:** 44 articles (40.0%)
- **Neutral:** 48 articles (43.6%)
- **Negative:** 18 articles (16.4%)

The dataset includes attributes such as publication date, news source/country, headline, text body, and sentiment label.

---

## 🤖 Model Performance & Architecture

- **Algorithm:** Logistic Regression (`max_iter=1000`, `random_state=42`)
- **Vectorizer:** TF-IDF (`ngram_range=(1, 2)`, `min_df=2`, `max_df=0.9`, `sublinear_tf=True`)
- **Evaluation Split:** 80% Train / 20% Test
- **Performance Metrics:** Evaluated using Weighted Precision, Recall, F1 Score, and Confusion Matrix.

---

## 📄 PDF Report Generation

The application includes an error-safe PDF generation engine powered by `fpdf2`:
- Uses project-local TTF fonts (`assets/fonts/arial.ttf`) for full Unicode compatibility (handles curly quotes, dashes, accented text, and quotes without encoding crashes).
- PDF reports are generated **lazily** upon clicking the download button.
- Includes analysis date, headline, article text snippet, predicted sentiment, confidence, probability breakdown, word count, character count, summary, and disclaimer.

---

## 🕘 Analysis History

- Personal history is stored per user account in `data/history.json`.
- Users can search saved records by keyword or filter by sentiment category.
- Supports deletion of individual records and direct PDF downloading for past analyses.

---

## ⚠️ Limitations

1. **Textual Tone Only:** The model evaluates language tone (positive/negative phrasing) and does not judge factual accuracy or truthfulness.
2. **Context Scope:** Satire, sarcasm, or highly nuanced political rhetoric may occasionally be misclassified.
3. **Dataset Size:** Trained on a demonstration corpus of 110 articles; expanding the dataset will further improve domain coverage.

---

## 🔮 Future Scope

- Integration with real-time news APIs (e.g., NewsAPI, GNews).
- Multi-language sentiment analysis for regional election news.
- Fine-tuned Deep Learning models (BERT, RoBERTa) for contextual transformers.
- Topic modeling and entity recognition for political candidates and parties.

---

## 📜 Disclaimer

This project is developed strictly for **educational and academic research purposes**. The sentiment predictions generated by this software represent statistical estimates based on natural language processing models. They do not represent political endorsements, factual verification, or predictions of election outcomes.
