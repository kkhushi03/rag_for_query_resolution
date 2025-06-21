import os, traceback, re, json, pickle
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.get_llm_func import rerank_results, llm_func, llm_gen_func
from utils.get_prompt_temp import prompt_retrieval_grader, prompt_generate_answer
from langchain.schema import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from rank_bm25 import BM25Okapi

# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHUNKS_OUT_PATH_TFIDF = Path(CONFIG["CHUNKS_OUT_PATH_TFIDF"])
TFIDF_DB_DIR = Path(CONFIG["TFIDF_DB_DIR"])
TFIDF_DB_PATH = Path(CONFIG["TFIDF_DB_PATH"])
TFIDF_META_PATH = Path(CONFIG["TFIDF_META_PATH"])

BATCH_SIZE = CONFIG["BATCH_SIZE"]
K = CONFIG["K"]
R = CONFIG["R"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
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

        documents = [
            Document(page_content=texts[i], metadata=metadata[i])
            for i in range(len(texts))
        ]

        logger.info("[BM25] Performing BM25 search...")
        tokenized_query = query_text.lower().split()
        scores = bm25_model.get_scores(tokenized_query)
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:at_k]
        results = [(documents[i], scores[i]) for i in top_k_idx]

        logger.info(f"[BM25] Retrieved {len(results)} documents")

        logger.info("[BM25] Reranking results with LLM...")
        reranked_results = rerank_results(query_text, results)
        reranked_results = reranked_results[:at_r]

        logger.info("[BM25] Grading results...")
        graded_results = []
        grader = prompt_retrieval_grader | llm_func | JsonOutputParser()
        for doc, score in reranked_results:
            try:
                grade = grader.invoke({"question": query_text, "document": doc.page_content})
                if grade["score"] == "YES":
                    graded_results.append((doc, score, grade["score"]))
            except Exception:
                continue

        if not graded_results:
            return []

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
            "rereranked_results": reranked_results,
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
            'rereranked_results': [],
            'graded_results': []
        }

def run_query_rag(query: str, tfidf_db_dir=TFIDF_DB_DIR, tfidf_db_path=TFIDF_DB_PATH, tfidf_metadata_path=TFIDF_META_PATH, at_k=K, at_r=R):
    logger = setup_logger("query_rag_logger", LOG_FILE)
    logger.info("++++++++Starting BM25 Query RAG Pipeline++++++++")
    return query_rag(query, tfidf_db_dir, tfidf_db_path, tfidf_metadata_path, at_k, at_r, logger)

if __name__ == "__main__":
    run_query_rag(query="Your default test query")
