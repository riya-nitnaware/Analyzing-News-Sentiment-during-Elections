import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    """
    Complete NLP preprocessing pipeline:
    1. Lowercasing
    2. URL Removal
    3. Punctuation/Special Character Removal
    4. Tokenization
    5. Stopword Removal
    6. Lemmatization
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Lowercasing
    text = text.lower()
    
    # 2. URL Removal
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 3. Punctuation/Special Character Removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Tokenization
    tokens = nltk.word_tokenize(text)
    
    # 5. Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens into a single string for TF-IDF
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text

def get_preprocessing_steps(text):
    """Returns intermediate steps for the Analyze News page"""
    if not isinstance(text, str):
        return {}
        
    steps = {}
    
    text_lower = text.lower()
    steps['Lowercasing'] = text_lower
    
    text_no_url = re.sub(r'http\S+|www\.\S+', '', text_lower)
    steps['URL Removal'] = text_no_url
    
    text_no_punct = text_no_url.translate(str.maketrans('', '', string.punctuation))
    text_no_punct = re.sub(r'[^a-zA-Z\s]', '', text_no_punct)
    steps['Punctuation Removal'] = text_no_punct
    
    tokens = nltk.word_tokenize(text_no_punct)
    steps['Tokenization'] = tokens
    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    steps['Stopword Removal'] = filtered_tokens
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    steps['Lemmatization'] = lemmatized_tokens
    
    steps['Final Cleaned Text'] = ' '.join(lemmatized_tokens)
    
    return steps
