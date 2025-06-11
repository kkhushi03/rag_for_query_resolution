import os, pickle, traceback
from pathlib import Path
from tqdm import tqdm
from utils.logger import setup_logger
from utils.config import CONFIG
from typing import List
from langchain.schema import Document
from utils.get_llm_func import embedding_func
from langchain_chroma import Chroma


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHROMA_DB_PATH = CONFIG["CHROMA_DB_PATH"]
EMBEDDINGS_OUT_PATH = Path(CONFIG["EMBEDDINGS_OUT_PATH"])
BATCH_SIZE = CONFIG["BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db/sub_stage_03_ingest_chroma.log")

def load_embedded_chunks(embeddings_dir: Path, logger) -> List[Document]:
    try:
        logger.info(f"Loading embedded chunks from {embeddings_dir}...")
        with open(embeddings_dir, "rb") as f:
            return pickle.load(f)
        logger.info(f"Loaded embedded chunks from {embeddings_dir} successfully.")
    except Exception as e:
        logger.error(f"Error loading embedded chunks: {e}")
        logger.debug(traceback.format_exc())
        return []

def run_chroma_ingest(embeddings_dir=EMBEDDINGS_OUT_PATH, chroma_dir=CHROMA_DB_PATH, batch_size=BATCH_SIZE):
    logger = setup_logger("chroma_logger", LOG_FILE)
    try:
        try:
            embedded_chunks = load_embedded_chunks(embeddings_dir)
            db = Chroma(
                embedding_function=embedding_func(), 
                persist_directory=chroma_dir
            )
            logger.info(f"Loaded Chroma DB at {chroma_dir}")

            existing_ids = set(db.get(include=[])["ids"])
            new_chunks = [e for e in embedded_chunks if e["id"] not in existing_ids]
        except Exception as e:
            logger.error(f"Error loading embedded chunks: {e}")
            logger.debug(traceback.format_exc())
            return

        try:
            logger.info(f"Total new unique chunks: {len(new_chunks)}")
            for i in tqdm(range(0, len(new_chunks), batch_size), desc="Ingesting"):
                batch = new_chunks[i:i + batch_size]
                db.add_documents(
                    documents=[Document(page_content=b["content"], metadata=b["metadata"]) for b in batch],
                    ids=[b["id"] for b in batch]
                )
            logger.info("Ingestion completed.")
        except Exception as e:
            logger.error(f"Error ingesting chunks: {e}")
            logger.debug(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error loading & ingesting chunks to DB: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    run_chroma_ingest()
