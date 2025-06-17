# rag_pipeline/stage_01_populate_tfidf.py
import os
import json
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.config import CONFIG
from utils.logger import setup_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.ingest_utils import load_documents_from_dirs
from langchain_core.documents import Document
# Config values
DATA_DIR = CONFIG["DATA_PATH"]
GROUPED_DIRS = CONFIG["GROUPED_DIRS"]
TFIDF_DB_PATH = CONFIG["TFIDF_DB_PATH"]
TFIDF_META_PATH = CONFIG["TFIDF_META_PATH"]
LOG_PATH = CONFIG["LOG_PATH"]

# Logging setup
log_dir = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "stage_01_populate_tfidf.log")
logger = setup_logger("tfidf_db_logger", log_file)


def run_populate_db():
    logger.info("Starting TF-IDF DB population...")
    documents = load_documents_from_dirs(DATA_DIR, GROUPED_DIRS)
    logger.info(f"Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["CHUNK_SIZE"],
        chunk_overlap=CONFIG["CHUNK_OVERLAP"]
    )

    all_docs = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        for idx, chunk in enumerate(chunks):
            raw_path = doc.metadata.get("source", "")
            group_number = Path(raw_path).parts[-2]

            metadata = {
                "source": raw_path,
                "chunk_id": f"{raw_path}_chunk_{idx}",
                "group_number": group_number
            }

            all_docs.append(Document(page_content=chunk, metadata=metadata))

    logger.info(f"Split into {len(all_docs)} chunks")

    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([doc.page_content for doc in all_docs])

    os.makedirs(os.path.dirname(TFIDF_DB_PATH), exist_ok=True)

    # ✅ Save Documents directly
    with open(TFIDF_DB_PATH, "wb") as f:
        pickle.dump((tfidf, vectors, all_docs), f)

    # Optionally save metadata separately (not required anymore)
    with open(TFIDF_META_PATH, "w", encoding="utf-8") as f:
        json.dump([doc.metadata for doc in all_docs], f, indent=2)

    logger.info(f"TF-IDF vectors and metadata saved to disk")
