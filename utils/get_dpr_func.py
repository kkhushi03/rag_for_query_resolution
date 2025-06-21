from typing import List
from utils.config import CONFIG
import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import warnings
from transformers import logging as transformers_logging


DPR_CONTEXT_ENCODER = CONFIG["DPR_CONTEXT_ENCODER"]
DPR_QUESTION_ENCODER = CONFIG["DPR_QUESTION_ENCODER"]

# Global DPR models - loaded once for efficiency
_dpr_question_encoder = None
_dpr_context_encoder = None
_dpr_question_tokenizer = None
_dpr_context_tokenizer = None

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*pooler.*")
warnings.filterwarnings("ignore", message=".*tokenizer class.*")
transformers_logging.set_verbosity_error()


def _load_dpr_models(dpr_q_encoder=DPR_QUESTION_ENCODER, dpr_c_encoder=DPR_CONTEXT_ENCODER):
    """Load DPR models and tokenizers once and cache them globally"""
    global _dpr_question_encoder, _dpr_context_encoder, _dpr_question_tokenizer, _dpr_context_tokenizer
    
    if _dpr_question_encoder is None:
        print("Loading DPR Question Encoder...")
        _dpr_question_encoder = DPRQuestionEncoder.from_pretrained(dpr_q_encoder)
        _dpr_question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_q_encoder)

    if _dpr_context_encoder is None:
        print("Loading DPR Context Encoder...")
        _dpr_context_encoder = DPRContextEncoder.from_pretrained(dpr_c_encoder)
        _dpr_context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_c_encoder)

class DPREmbeddings:
    """Custom DPR embeddings class that mimics LangChain's embedding interface"""
    
    def __init__(self):
        _load_dpr_models()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move models to device
        _dpr_question_encoder.to(self.device)
        _dpr_context_encoder.to(self.device)
    
    def __call__(self, text: str) -> List[float]:
        """Make the class callable - FAISS compatibility"""
        return self.embed_query(text)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using DPR question encoder"""
        with torch.no_grad():
            inputs = _dpr_question_tokenizer(
                text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True, 
                padding=True
            ).to(self.device)
            
            outputs = _dpr_question_encoder(**inputs)
            embeddings = outputs.pooler_output
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy().flatten().tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using DPR context encoder"""
        embeddings_list = []
        
        # Process in batches to avoid memory issues
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = _dpr_context_tokenizer(
                    batch, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True, 
                    padding=True
                ).to(self.device)
                
                outputs = _dpr_context_encoder(**inputs)
                batch_embeddings = outputs.pooler_output
                
                # Normalize embeddings
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings_list.extend(batch_embeddings.cpu().numpy().tolist())
        
        return embeddings_list