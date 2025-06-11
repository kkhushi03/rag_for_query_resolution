import os, json, pickle, gc, traceback
from pathlib import Path
from tqdm import tqdm
from utils.logger import setup_logger
from utils.config import CONFIG
from typing import List
from langchain.schema import Document
from utils.get_llm_func import embedding_func, num_tokens


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHUNKS_OUT_PATH = Path(CONFIG["CHUNKS_OUT_PATH"])
EMBEDDINGS_OUT_PATH = Path(CONFIG["EMBEDDINGS_OUT_PATH"])
CHUNK_LOWER_LIMIT = CONFIG["CHUNK_LOWER_LIMIT"]
CHUNK_UPPER_LIMIT = CONFIG["CHUNK_UPPER_LIMIT"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db/sub_stage_02_embed_chunks.log")


def load_chunks(chunk_dir: Path, logger):
    try:
        logger.info(f"Loading chunks from {chunk_dir}...")
        with open(chunk_dir, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                yield obj
        logger.info(f"Loaded chunks from {chunk_dir} successfully.")
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        logger.debug(traceback.format_exc())

def filter_and_embed(chunks: List[Document], lower_limit: int, upper_limit: int, batch_size: int, logger) -> List[Document]:
    try:
        contents, metas, ids = [], [], []
        logger.info(f"Applying custom (token-based) filtering on chunks before ingestion...")
        for chunk in chunks:
            content = chunk["content"].strip()
            meta = chunk["metadata"]
            tid = meta.get("chunk_id", None)
            if lower_limit < num_tokens(content) < upper_limit:
                contents.append(content)
                metas.append(meta)
                ids.append(tid)
        logger.info(f"Chunks remaining before filter: {len(chunks)}")
        logger.info(f"Chunks remaining after filter: {len(contents)}")
    except Exception as e:
        logger.error(f"Error applying custom filtering on chunks: {e}")
        logger.debug(traceback.format_exc())
        return

    try:
        embeddings = []
        logger.info(f"Embedding {len(contents)} chunks...")
        for i in tqdm(range(0, len(contents), batch_size), desc="Embedding"):
            batch = contents[i:i + batch_size]
            embeddings.extend(embedding_func().embed_documents(batch))
        logger.info(f"Embedded {len(embeddings)} chunks successfully.")

        logger.info(f"Serializing {len(embeddings)} embedded chunks...")
        embedded = [{"embedding": e, "content": c, "metadata": m, "id": i}
                    for e, c, m, i in zip(embeddings, contents, metas, ids)]
        logger.info(f"Serialized {len(embedded)} embedded chunks successfully.")
        return embedded
    except Exception as e:
        logger.error(f"Error embedding chunks: {e}")
        logger.debug(traceback.format_exc())
        return

def run_embed_chunks(chunk_dir=CHUNKS_OUT_PATH, lower_limit=CHUNK_LOWER_LIMIT, upper_limit=CHUNK_UPPER_LIMIT, batch_size=BATCH_SIZE, embeddings_dir=EMBEDDINGS_OUT_PATH):
    logger = setup_logger("embed_logger", LOG_FILE)
    try:
        chunks = list(load_chunks(chunk_dir, logger))
        embedded = filter_and_embed(chunks, lower_limit, upper_limit, batch_size, logger=logger)
        
        embeddings_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(embeddings_dir, "wb") as f:
            pickle.dump(embedded, f)
        logger.info(f"Saved {len(embedded)} embedded chunks to {embeddings_dir}")
    except Exception as e:
        logger.error(f"Error embedding chunks: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    run_embed_chunks()
