import os, json, pickle, shutil, gc, traceback, re
from tqdm import tqdm
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from utils.config import CONFIG
from utils.logger import setup_logger
from utils.get_llm_func import num_tokens, rerank_results, llm_func, llm_gen_func
from utils.get_prompt_temp import prompt_retrieval_grader, prompt_generate_answer

LOG_PATH = Path(CONFIG["LOG_PATH"])
DATA_PATH = Path(CONFIG["DATA_PATH"])
GROUPED_DIRS = CONFIG["GROUPED_DIRS"]
CHUNKS_OUT_PATH_BM25 = Path(CONFIG["CHUNKS_OUT_PATH_TFIDF"])
TFIDF_DB_DIR = Path(CONFIG["TFIDF_DB_DIR"])
TFIDF_DB_PATH = Path(CONFIG["TFIDF_DB_PATH"])
TFIDF_META_PATH = Path(CONFIG["TFIDF_META_PATH"])

CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_LOWER_LIMIT = CONFIG["CHUNK_LOWER_LIMIT"]
CHUNK_UPPER_LIMIT = CONFIG["CHUNK_UPPER_LIMIT"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
INGEST_BATCH_SIZE = CONFIG["INGEST_BATCH_SIZE"]
K = CONFIG.get("K", 5)
R = CONFIG.get("R", 5)

LOG_DIR = os.path.join(os.getcwd(), str(LOG_PATH))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_02_query_rag.log")

def query_rag(query_text: str, tfidf_db_dir, tfidf_db_path, tfidf_metadata_path, at_k, at_r, logger):
    try:
        if not tfidf_db_dir.exists():
            logger.error(f"BM25 DB not found at path: {tfidf_db_dir}. Please populate the DB first.")
            return []

        with open(tfidf_db_path, "rb") as f:
            bm25_model, tokenized_corpus, texts = pickle.load(f)

        with open(tfidf_metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, list):
            logger.error("[BM25] Metadata JSON is not a list.")
            return []

        documents = []
        for i in range(len(texts)):
            content = texts[i]
            meta = metadata[i]

            if isinstance(content, dict) and isinstance(meta, str):
                content, meta = meta, content

            if not isinstance(content, str):
                logger.warning(f"[BM25] Skipping chunk with invalid content at index {i}")
                continue

            if not isinstance(meta, dict):
                logger.warning(f"[BM25] Metadata at index {i} is not a dict, using empty dict")
                meta = {}

            documents.append(Document(page_content=content, metadata=meta))

        logger.info("[BM25] Performing BM25 search...")
        tokenized_query = query_text.lower().split()
        logger.debug(f"[BM25] Tokenized query: {tokenized_query}")

        scores = bm25_model.get_scores(tokenized_query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:at_k]
        results = [(documents[i], scores[i]) for i in top_k_idx]
        logger.info(f"[BM25] Retrieved {len(results)} documents")
        logger.debug(f"[BM25] Top BM25 scores: {[scores[i] for i in top_k_idx]}")

        logger.debug("[BM25] Preview of top retrieved chunks:")
        for doc, score in results:
            logger.debug(f"Score: {score:.4f}, Preview: {doc.page_content[:300]}")

        logger.info("[BM25] Reranking results with LLM...")
        reranked_results = rerank_results(query_text, results)
        reranked_results = reranked_results[:at_r]

        # Optional: sort reranked results if not already sorted
        reranked_results.sort(key=lambda x: x[1], reverse=True)

        logger.info("[BM25] Grading results...")
        graded_results = [(doc, score, "YES") for doc, score in reranked_results]

        if not graded_results:
            logger.warning("[BM25] No relevant results found after reranking.")
            return {
                "llms_response": "No relevant documents found.",
                "context": "",
                "sources": [],
                "results": results,
                "reranked_results": [],
                "graded_results": []
            }

        logger.info("[BM25] Generating final answer...")
        context = "\n\n---\n\n".join([doc.page_content for doc, _, _ in graded_results])
        response = prompt_generate_answer | llm_gen_func | StrOutputParser()
        llm_answer = response.invoke({"question": query_text, "context": context})

        sources = [doc.metadata.get("chunk_id") for doc, _, _ in graded_results]

        return {
            "llms_response": llm_answer,
            "context": context,
            "sources": sources,
            "results": results,
            "reranked_results": reranked_results,
            "graded_results": graded_results
        }

    except Exception as e:
        logger.error(f"[BM25] Error during RAG pipeline: {e}")
        logger.debug(traceback.format_exc())
        return {
            'llms_response': '',
            'context': '',
            'sources': [],
            'results': [],
            'reranked_results': [],
            'graded_results': []
        }

def run_query_rag(query: str, tfidf_db_dir=TFIDF_DB_DIR, tfidf_db_path=TFIDF_DB_PATH, tfidf_metadata_path=TFIDF_META_PATH, at_k=K, at_r=R):
    logger = setup_logger("query_rag_logger", LOG_FILE)
    logger.info("++++++++Starting BM25 Query RAG Pipeline++++++++")
    return query_rag(query, tfidf_db_dir, tfidf_db_path, tfidf_metadata_path, at_k, at_r, logger)

if __name__ == "__main__":
    run_query_rag(query="Your default test query")