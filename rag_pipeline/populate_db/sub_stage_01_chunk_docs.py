import os, json, gc, traceback
from pathlib import Path
from tqdm import tqdm
from utils.logger import setup_logger
from utils.config import CONFIG
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
DATA_PATH = Path(CONFIG["DATA_PATH"])
GROUPED_DIRS = CONFIG["GROUPED_DIRS"]
CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
CHUNKS_OUT_PATH = Path(CONFIG["CHUNKS_OUT_PATH"])

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db/sub_stage_01_chunk_docs.log")


def load_pdfs(grouped_dirs: List[str], data_dir: Path, logger) -> List[Document]:
    all_docs = []
    logger.info(f"Loading PDFs from {grouped_dirs}.....")
    for group in grouped_dirs:
        group_path = data_dir / group
        if not group_path.exists(): continue
        pdfs = list(group_path.rglob("*.pdf"))
        logger.info(f"Found {len(pdfs)} PDFs in {group}")
        for file in tqdm(pdfs, desc=f"Loading from {group}"):
            try:
                loader = PyMuPDFLoader(str(file))
                docs = loader.load()
                all_docs.extend(docs)
                logger.debug(f"Loaded {len(docs)} from {file}")
            except Exception as e:
                logger.warning(f"Failed to load {file}: {e}")
                continue
            finally:
                gc.collect()
    return all_docs

def split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int, logger) -> List[Document]:
    try:
        logger.info(f"Splitting {len(docs)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.info(f"Splitted into {len(splitter.split_documents(docs))} chunks")
        return splitter.split_documents(docs)
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def serialize_chunks(chunks, chunk_dir, logger):
    try:
        logger.info(f"Serializing {len(chunks)} chunks to {chunk_dir}...")
        chunk_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(chunk_dir, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps({"content": chunk.page_content, "metadata": chunk.metadata}) + "\n")
                logger.debug(f"Serialized chunk: {chunk}")
        logger.info(f"Serialized {len(chunks)} chunks to {chunk_dir} successfully.")
    except Exception as e:
        logger.error(f"Error serializing chunks: {e}")
        logger.debug(traceback.format_exc())

def run_chunk_docs(grouped_dirs=GROUPED_DIRS, data_dir=DATA_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, chunk_dir=CHUNKS_OUT_PATH, logger=LOG_FILE):
    logger = setup_logger("chunk_logger", LOG_FILE)
    try:
        docs = load_pdfs(grouped_dirs, data_dir, logger)
        if not docs:
            logger.error("No docs loaded. Exiting.")
            return
        
        chunks = split_docs(docs, chunk_size, chunk_overlap, logger)
        if not chunks:
            logger.error("No chunks created. Exiting.")
            return
        
        logger.info(f"Loaded {len(docs)} docs, created {len(chunks)} chunks")
        serialize_chunks(chunks, chunk_dir, logger)
        logger.info(f"Chunks saved to {chunk_dir}")
    except Exception as e:
        logger.error(f"Error running chunk_docs: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    run_chunk_docs()
