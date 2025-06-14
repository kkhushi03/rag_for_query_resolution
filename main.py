# main.py (Updated for TF-IDF-based RAG pipeline)
import os
from rag_pipeline.stage_01_populate_tfidf import run_populate_db
from rag_pipeline.stage_02_query_data import run_query_rag
from utils.config import CONFIG

if __name__ == "__main__":
    print("[MAIN] Running TF-IDF RAG pipeline...")

    if not os.path.exists(CONFIG["TFIDF_DB_PATH"]):
        print("[MAIN] Populating TF-IDF DB from documents...")
        run_populate_db()
    else:
        print("[MAIN] Skipping DB population, using existing TF-IDF index.")

    query = CONFIG["QUERY_TEXT"]
    print(f"[MAIN] Running query: {query}")
    run_query_rag(query)
    print("[MAIN] Pipeline execution complete.")
