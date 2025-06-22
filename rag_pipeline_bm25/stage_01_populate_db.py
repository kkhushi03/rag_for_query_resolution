import os, json, pickle, shutil, gc, traceback
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils.get_population_func import (
    load_docs, flatten_docs, split_docs, calc_chunk_ids, load_existing_chunks_metadata, filter_chunks, process_in_batches, save_processed_chunks_metadata
)
from langchain.schema import Document


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
DATA_PATH = Path(CONFIG["DATA_PATH"])
GROUPED_DIRS = CONFIG["GROUPED_DIRS"]

CHUNKS_OUT_PATH_BM25 = Path(CONFIG["CHUNKS_OUT_PATH_BM25"])
BM25_DB_DIR = Path(CONFIG["BM25_DB_DIR"])
BM25_DB_PATH = Path(CONFIG["BM25_DB_PATH"])
BM25_META_PATH = Path(CONFIG["BM25_META_PATH"])

CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_LOWER_LIMIT = CONFIG["CHUNK_LOWER_LIMIT"]
CHUNK_UPPER_LIMIT = CONFIG["CHUNK_UPPER_LIMIT"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
INGEST_BATCH_SIZE = CONFIG["INGEST_BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db.log")

# Download required NLTK data (run once)
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')


def save_to_bm25_db(chunks: list[Document], bm25_db_path, bm25_metadata_path, base_data_path, chunks_dir, lower_limit, upper_limit, ingest_batch_size, logger):
    try:
        logger.info("[Stage 01, Part 06] Saving chunks to BM25 DB.....")

        # calculate "page:chunk" IDs
        # Step 1: Assign chunk IDs
        chunks_with_ids = calc_chunk_ids(chunks, base_data_path, logger)
        logger.info(f"[Stage 01, Part 06.1] Calculated chunk IDs for total {len(chunks_with_ids)} chunks")
        
        # add/update the docs
        # Step 2: Load existing chunk IDs from metadata file (not from FAISS directly)
        existing_ids = load_existing_chunks_metadata(chunks_dir, logger)
        logger.info(f"[Stage 01, Part 06.2] No. of existing chunk IDs tracked: {len(existing_ids)}")
        
        # only add docs that don't exist in the db
        # Step 3: Filter only new chunks
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["chunk_id"] not in existing_ids:
                new_chunks.append(chunk)
        logger.info(f"[Stage 01, Part 06.3] No. of new chunks to add: {len(new_chunks)}")
        
        # Step 4: Safety Net — remove duplicates within `new_chunks` -> Deduplicate
        seen_ids = set()
        unique_new_chunks = []
        for chunk in new_chunks:
            cid = chunk.metadata["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_new_chunks.append(chunk)
        logger.info(f"[Stage 01, Part 06.4] No. of unique new chunks to add, after deduplication: {len(unique_new_chunks)}")
        
        if not unique_new_chunks:
            logger.info("[Stage 01, Part 06.4] No new unique chunks to process. Database is up to date.")
            return
        
        # Step 5: Filter out chunks outside token limits (especially empty content chunks)
        filtered_chunks = filter_chunks(unique_new_chunks, lower_limit, upper_limit, logger)
        if not filtered_chunks:
            logger.info("[Stage 01, Part 06.5] No valid (non-empty) chunks remaining after filtering")
            return
        
        # Step 6: Collect text & metadata in batches
        texts = []
        metadatas = []

        def collect_batch(batch):
            for doc in batch:
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)

        process_in_batches(
            documents=filtered_chunks,
            ingest_batch_size=ingest_batch_size,
            ingest_fn=collect_batch,
            logger=logger
        )
        logger.info(f"[Stage 01, Part 07] Collected {len(texts)} texts and {len(metadatas)} metadata entries from chunks")

        # Step 7: Vectorize using BM25
        logger.info(f"[Stage 01, Part 08] Vectorizing {len(texts)} texts using BM25...")
        try:
            stop_words = set(stopwords.words('english'))
            tokenized_texts = []
            for text in texts:
                tokens = word_tokenize(text.lower())
                tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
                tokenized_texts.append(tokens)
            
            # Create BM25 index
            bm25 = BM25Okapi(tokenized_texts)
            logger.info(f"[Stage 01, Part 08] Vectorization complete. Shape of BM25 matrix: {len(tokenized_texts)}")
        except Exception as e:
            logger.error(f"[Stage 01, Part 08] Error during BM25 vectorization: {e}")
            logger.debug(traceback.format_exc())
            return

        # Step 8: Save BM25 index and metadata
        logger.info(f"[Stage 01, Part 09] Saving BM25 index to {bm25_db_path} and metadata to {bm25_metadata_path}...")
        try:
            os.makedirs(os.path.dirname(bm25_db_path), exist_ok=True)
            with open(bm25_db_path, "wb") as f:
                pickle.dump((bm25, texts, tokenized_texts), f)

            with open(bm25_metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadatas, f, indent=2)
            
            logger.info(f"[Stage 01, Part 09] BM25 index and metadata saved successfully")
        except Exception as e:
            logger.error(f"[Stage 01, Part 09] Error saving BM25 index or metadata: {e}")
            logger.debug(traceback.format_exc())
            return

        # Step 9: Save processed metadata
        logger.info(f"[Stage 01, Part 10] Saving processed chunks metadata to {chunks_dir}...")
        reconstructed_docs = [
            Document(page_content=texts[i], metadata=metadatas[i])
            for i in range(len(texts))
        ]
        save_processed_chunks_metadata(reconstructed_docs, chunks_dir, logger)
        
        logger.info("[Stage 01, Part 10] Chunks saved to BM25 DB successfully")
    except Exception as e:
        logger.error(f"[Stage 01, Part 10] Error saving to BM25 DB: {e}")
        logger.debug(traceback.format_exc())
        return []

def clear_database(bm25_db_dir):
    if os.path.exists(bm25_db_dir):
        shutil.rmtree(bm25_db_dir)

def clear_chunks(chunks_dir):
    chunks_path = Path(chunks_dir)
    if chunks_path.exists():
        if chunks_path.is_file():
            chunks_path.unlink()
        elif chunks_path.is_dir():
            shutil.rmtree(chunks_path)

def run_populate_db(reset=False, bm25_db_dir=BM25_DB_DIR, bm25_db_path=BM25_DB_PATH, bm25_metadata_path=BM25_META_PATH, base_data_path=DATA_PATH, chunks_dir=CHUNKS_OUT_PATH_BM25, grouped_dirs=GROUPED_DIRS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, lower_limit=CHUNK_LOWER_LIMIT, upper_limit=CHUNK_UPPER_LIMIT, ingest_batch_size=INGEST_BATCH_SIZE):
    try:
        logger = setup_logger("populate_db_logger", LOG_FILE)
        logger.info(" ")
        logger.info("--------++++++++Starting db population stage.....")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("[Stage 01, Part 00.1] (RESET DB) Clearing the database...")
            clear_database(bm25_db_dir)
            logger.info("[Stage 01, Part 00.1] (RESET DB) Database cleared successfully.")
            
            logger.info("[Stage 01, Part 00.2] (RESET DB) Clearing the chunks file...")
            clear_chunks(chunks_dir)
            logger.info("[Stage 01, Part 00.2] (RESET DB) Chunks file cleared successfully.")
        
        # create (or update) the db
        docs = load_docs(base_data_path, grouped_dirs, logger)
        if not docs:
            logger.error("[Stage 01] No docs loaded. Exiting.")
            return
        
        flat_docs = flatten_docs(docs, logger)
        
        chunks = split_docs(flat_docs, chunk_size, chunk_overlap, logger)
        if not chunks:
            logger.error("[Stage 01] No chunks created. Exiting.")
            return
        
        logger.debug(f" [] Loaded {len(flat_docs)} docs, created {len(chunks)} chunks")
        
        save_to_bm25_db(chunks, bm25_db_path, bm25_metadata_path, base_data_path, chunks_dir, lower_limit, upper_limit, ingest_batch_size, logger)
        
        # Manual memory cleanup
        del docs, flat_docs, chunks
        gc.collect()
        
        logger.info("--------++++++++DB population stage successfully completed.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error at [Stage 01]: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_populate_db()