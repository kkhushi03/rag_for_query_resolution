from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenize_text(text):
    """Tokenize and preprocess text for BM25"""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]

def create_bm25_index(texts):
    """Create BM25 index from list of texts"""
    tokenized_texts = [tokenize_text(text) for text in texts]
    return BM25Okapi(tokenized_texts), tokenized_texts