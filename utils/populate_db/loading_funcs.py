import json, traceback, pickle
from pathlib import Path
from typing import List
from langchain.schema import Document

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