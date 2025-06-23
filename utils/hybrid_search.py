import nltk, os
import numpy as np
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langchain.schema import Document
from sklearn.preprocessing import MinMaxScaler


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
BM25_WEIGHT = CONFIG["BM25_WEIGHT"]
FAISS_WEIGHT = CONFIG["FAISS_WEIGHT"]
HYBRID_TOP_K = CONFIG["HYBRID_TOP_K"]
RRF_PARAM = CONFIG["RRF_PARAM"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "hybrid_utils.log")
logger = setup_logger("hybrid_search", LOG_FILE)


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class HybridRetriever:
    def __init__(self, bm25_weight=BM25_WEIGHT, faiss_weight=FAISS_WEIGHT):
        """
        Initialize hybrid retriever with configurable weights.
        
        Args:
            bm25_weight (float): Weight for BM25 scores (0-1)
            faiss_weight (float): Weight for FAISS scores (0-1)
        """
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        self.bm25_index = None
        self.faiss_db = None
        self.documents = None
        self.tokenized_corpus = None
        
        logger.info("Initializing preprocessing tools...")
        try:
            logger.debug("Loading NLTK stopwords...")
            self.stop_words = set(stopwords.words('english'))
            logger.debug("NLTK stopwords loaded successfully.")
        except LookupError:
            logger.warning("NLTK stopwords not found. Downloading...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        logger.info("Preprocessing tools initialized successfully.")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text for BM25.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            List[str]: List of preprocessed tokens
        """
        try:
            logger.debug("Tokenizing text...")
            if not text:
                logger.error("Empty text provided for preprocessing.")
                return []
            tokens = word_tokenize(text.lower())
            
            logger.debug("Filtering tokens (keep only alphanumeric tokens, remove stopwords)...")
            if not tokens:
                logger.error("No tokens found after tokenization.")
                return []
            tokens = [
                token for token in tokens 
                if token.isalnum() and token not in self.stop_words and len(token) > 1
            ]
            
            logger.debug(f"Preprocessed {len(tokens)} tokens from text.")
            return tokens
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            logger.warning("Falling back to simple split for preprocessing.")
            return [word.lower() for word in text.split() if len(word) > 1]
    
    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents (List[Document]): List of documents to index
        """
        try:
            logger.debug(f"Building BM25 index for {len(documents)} documents for later retrieval...")
            self.documents = documents
            
            logger.debug("Tokenizing documents for BM25 index...")
            self.tokenized_corpus = []
            for doc in documents:
                tokens = self.preprocess_text(doc.page_content)
                self.tokenized_corpus.append(tokens)
            
            logger.debug("Building BM25 index...")
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
            logger.debug("BM25 index built successfully!")
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            raise ValueError("Failed to build BM25 index. Check logs for details.")
    
    def bm25_search(self, query: str, top_k: int = HYBRID_TOP_K) -> List[Tuple[Document, float]]:
        """
        Perform BM25 search.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        try:
            if self.bm25_index is None or self.documents is None:
                logger.error("BM25 index not built. Call build_bm25_index() first.")
                raise ValueError("BM25 index not built. Call build_bm25_index() first.")
            
            logger.info(f"Preprocessing/Tokenizing query: {query}")
            query_tokens = self.preprocess_text(query)
            if not query_tokens:
                logger.error("No valid tokens found in query. Returning empty results.")
                return []
            
            logger.info(f"Calculating BM25 scores for query: {query_tokens}")
            scores = self.bm25_index.get_scores(query_tokens)
            
            logger.info(f"Selecting top {top_k} results based on BM25 scores...")
            if len(scores) < top_k:
                logger.warning(f"Requested top_k ({top_k}) exceeds available documents ({len(scores)}). Adjusting top_k.")
                top_k = len(scores)
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            logger.info(f"Returning top {top_k} BM25 results...")
            results = [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            logger.info(f"BM25 search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def faiss_search(self, faiss_db, query: str, top_k: int = HYBRID_TOP_K) -> List[Tuple[Document, float]]:
        """
        Perform FAISS similarity search.
        
        Args:
            faiss_db: FAISS database instance
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        try:
            # FAISS similarity search returns (Document, similarity_score) tuples
            logger.info(f"Performing FAISS search for query: {query}")
            if not faiss_db:
                logger.error("FAISS database not initialized. Please provide a valid FAISS instance.")
                raise ValueError("FAISS database not initialized. Please provide a valid FAISS instance.")
            
            results = faiss_db.similarity_search_with_score(query, k=top_k)
            logger.info(f"FAISS search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            print(f"Error in FAISS search: {e}")
            return []
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to 0-1 range using Min-Max scaling.
        
        Args:
            scores (List[float]): Raw scores
            
        Returns:
            List[float]: Normalized scores
        """
        try:
            if not scores:
                logger.warning("Empty scores list provided for normalization.")
                return []
            
            scores_array = np.array(scores).reshape(-1, 1)
            
            logger.info("Normalizing scores using Min-Max scaling...")
            if np.max(scores_array) == np.min(scores_array):
                return [1.0] * len(scores)
            
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(scores_array).flatten()
            
            logger.info("Scores normalized successfully.")
            logger.info(f"Converting normalized scores to list.....")
            return normalized.tolist()
        except Exception as e:
            logger.error(f"Error normalizing scores: {e}")
            return [0.0] * len(scores)
    
    def reciprocal_rank_fusion(self, bm25_results: List[Tuple[Document, float]], faiss_results: List[Tuple[Document, float]], rf_param: int = RRF_PARAM) -> List[Tuple[Document, float]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            bm25_results: BM25 search results
            faiss_results: FAISS search results  
            rf_param: RRF parameter (default 60)
            
        Returns:
            List[Tuple[Document, float]]: Fused and ranked results
        """
        try:
            logger.info("Combining BM25 and FAISS results using Reciprocal Rank Fusion (RRF)...")
            if not bm25_results and not faiss_results:
                logger.error("Both BM25 and FAISS results are empty. Returning empty results.")
                return []
            doc_scores = {}
            
            logger.info("Processing BM25 results for RRF...")
            for rank, (doc, score) in enumerate(bm25_results):
                chunk_id = doc.metadata.get('chunk_id', id(doc))
                rrf_score = 1.0 / (rf_param + rank + 1)
                doc_scores[chunk_id] = doc_scores.get(chunk_id, {'doc': doc, 'score': 0}) 
                doc_scores[chunk_id]['score'] += self.bm25_weight * rrf_score
            
            logger.info("Processing FAISS results for RRF...")
            for rank, (doc, score) in enumerate(faiss_results):
                chunk_id = doc.metadata.get('chunk_id', id(doc))
                rrf_score = 1.0 / (rf_param + rank + 1)
                if chunk_id not in doc_scores:
                    doc_scores[chunk_id] = {'doc': doc, 'score': 0}
                doc_scores[chunk_id]['score'] += self.faiss_weight * rrf_score
            
            logger.info("Sorting fused results by combined score...")
            fused_results = [(item['doc'], item['score']) for item in doc_scores.values()]
            fused_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"RRF fusion completed. Returning {len(fused_results)} results.")
            return fused_results
        except Exception as e:
            logger.error(f"Error in Reciprocal Rank Fusion: {e}")
            return []
    
    def weighted_score_fusion(self, bm25_results: List[Tuple[Document, float]], faiss_results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Combine results using weighted score fusion.
        
        Args:
            bm25_results: BM25 search results
            faiss_results: FAISS search results
            
        Returns:
            List[Tuple[Document, float]]: Fused and ranked results
        """
        try:
            logger.info("Extracting scores for normalization...")
            logger.info("Now combining BM25 and FAISS results using weighted score fusion...")
            bm25_scores = [score for _, score in bm25_results]
            faiss_scores = [score for _, score in faiss_results]
            if not bm25_scores and not faiss_scores:
                logger.error("Both BM25 and FAISS results are empty. Returning empty results.")
                return []
            
            logger.info("Normalizing BM25 and FAISS scores...")
            bm25_normalized = self.normalize_scores(bm25_scores)
            faiss_normalized = self.normalize_scores(faiss_scores)
            
            logger.info("Creating document to score mapping for fusion...")
            doc_scores = {}
            
            logger.info("Adding normalized BM25 scores to document scores...")
            for i, (doc, _) in enumerate(bm25_results):
                chunk_id = doc.metadata.get('chunk_id', id(doc))
                doc_scores[chunk_id] = {
                    'doc': doc, 
                    'bm25_score': bm25_normalized[i] if i < len(bm25_normalized) else 0,
                    'faiss_score': 0
                }
            
            logger.info("Adding normalized FAISS scores to document scores...")
            for i, (doc, _) in enumerate(faiss_results):
                chunk_id = doc.metadata.get('chunk_id', id(doc))
                faiss_score = faiss_normalized[i] if i < len(faiss_normalized) else 0
                
                if chunk_id in doc_scores:
                    doc_scores[chunk_id]['faiss_score'] = faiss_score
                else:
                    doc_scores[chunk_id] = {
                        'doc': doc,
                        'bm25_score': 0,
                        'faiss_score': faiss_score
                    }
            
            logger.info("Calculating combined scores using weighted fusion...")
            fused_results = []
            for chunk_id, data in doc_scores.items():
                combined_score = (
                    self.bm25_weight * data['bm25_score'] + 
                    self.faiss_weight * data['faiss_score']
                )
                fused_results.append((data['doc'], combined_score))
            
            logger.info("Sorting fused results by combined score...")
            fused_results.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Weighted score fusion completed. Returning {len(fused_results)} results.")
            return fused_results
        except Exception as e:
            logger.error(f"Error in weighted score fusion: {e}")
            return []
    
    def hybrid_search(self, faiss_db, query: str, top_k: int = HYBRID_TOP_K, fusion_method: str = "rrf") -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining BM25 and FAISS.
        
        Args:
            faiss_db: FAISS database instance
            query (str): Search query
            top_k (int): Number of final results to return
            fusion_method (str): "rrf" for Reciprocal Rank Fusion or "weighted" for weighted fusion
            
        Returns:
            List[Tuple[Document, float]]: Hybrid search results
        """
        try:
            logger.info(f"Starting hybrid search with query: {query}, top_k: {top_k}, fusion_method: {fusion_method}")
            if fusion_method not in ["rrf", "weighted"]:
                logger.error(f"Invalid fusion method: {fusion_method}. Must be 'rrf' or 'weighted'.")
                raise ValueError("Invalid fusion method. Must be 'rrf' or 'weighted'.")
            search_k = max(top_k * 3, 50)
            
            logger.info("Performing BM25 search...")
            bm25_results = self.bm25_search(query, top_k=search_k)
            
            # Perform FAISS search 
            logger.info("Performing FAISS search...")
            faiss_results = self.faiss_search(faiss_db, query, top_k=search_k)
            
            logger.info(f"Combining results using {fusion_method} method...")
            if fusion_method == "rrf":
                logger.info("Using Reciprocal Rank Fusion method...")
                fused_results = self.reciprocal_rank_fusion(bm25_results, faiss_results)
            else:
                logger.info("Using weighted score fusion method...")
                fused_results = self.weighted_score_fusion(bm25_results, faiss_results)
            
            logger.info(f"Returning top {top_k} results from hybrid search...")
            return fused_results[:top_k]
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []