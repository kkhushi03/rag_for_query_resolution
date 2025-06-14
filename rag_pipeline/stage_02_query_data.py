# rag_pipeline/stage_02_query_data.py
import os
import json
import pickle
import traceback
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.get_llm_func import rerank_results, llm_func, llm_gen_func
from utils.get_prompt_temp import prompt_retrieval_grader, prompt_generate_answer
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document

# Configs
LOG_PATH = CONFIG["LOG_PATH"]
TFIDF_DB_PATH = CONFIG["TFIDF_DB_PATH"]
TFIDF_META_PATH = CONFIG["TFIDF_META_PATH"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]
K = CONFIG["K"]
R = CONFIG["R"]

# Setup logger
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_02_query_rag.log")

from langchain_core.documents import Document

def query_rag(query_text: str, logger):
    try:
        logger.info("Loading TF-IDF DB...")
        with open(TFIDF_DB_PATH, "rb") as f:
            tfidf, vectors, texts = pickle.load(f)

        with open(TFIDF_META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        query_vector = tfidf.transform([query_text])
        sims = cosine_similarity(query_vector, vectors).flatten()
        top_k_idx = sims.argsort()[-K:][::-1]

        retrieved_docs = []
        for i in top_k_idx:
            doc = Document(page_content=texts[i], metadata=metadata[i])
            retrieved_docs.append((doc, sims[i]))

        logger.info(f"Retrieved top-{K} documents")

        # Grading directly on Document objects
        graded_results = []
        grader = prompt_retrieval_grader | llm_func | JsonOutputParser()

        for i, (doc, score) in enumerate(retrieved_docs[:R]):
            try:
                grade_input = {"question": query_text, "document": doc.page_content}
                grade = grader.invoke(grade_input)

                if isinstance(grade, dict) and grade.get("score") == "YES":
                    graded_results.append((doc, score, grade["score"]))
            except Exception as e:
                logger.error(f"[Grading Error] Doc {i}: {e}")
                logger.debug(traceback.format_exc())

        if not graded_results:
            logger.warning("No documents passed grading")
            return {}

        # Generate answer
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _, _ in graded_results])
        generator = prompt_generate_answer | llm_gen_func | StrOutputParser()
        response = generator.invoke({"question": query_text, "context": context_text})

        sources = [doc.metadata.get("chunk_id") for doc, _, _ in graded_results]

        return {
            "llms_response": response,
            "context": context_text,
            "sources": sources,
            "graded_results": graded_results
        }

    except Exception as e:
        logger.error(f"Query RAG failed: {e}")
        logger.debug(traceback.format_exc())
        return {}

def run_query_rag(query: str):
    logger = setup_logger("query_rag_logger", LOG_FILE)
    logger.info("Running TF-IDF query...")
    return query_rag(query, logger)
