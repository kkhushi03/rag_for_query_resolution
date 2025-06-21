import os, shutil, gc, traceback
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.get_population_func import (
    load_docs, flatten_docs, split_docs, calc_chunk_ids, filter_chunks, process_in_batches
)
from utils.get_llm_func import embedding_func
from langchain.schema import Document
from langchain_chroma import Chroma


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
DATA_PATH = Path(CONFIG["DATA_PATH"])
GROUPED_DIRS = CONFIG["GROUPED_DIRS"]
CHROMA_DB_PATH = CONFIG["CHROMA_DB_PATH"]

CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_LOWER_LIMIT = CONFIG["CHUNK_LOWER_LIMIT"]
CHUNK_UPPER_LIMIT = CONFIG["CHUNK_UPPER_LIMIT"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
INGEST_BATCH_SIZE = CONFIG["INGEST_BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db.log")


def save_to_chroma_db(chunks: list[Document], chroma_db_dir, base_data_path, lower_limit, upper_limit, ingest_batch_size, logger):
    try:
        logger.info("[Stage 01, Part 06] Saving chunks to Chroma DB.....")
        
        # load the existing db
        db = Chroma(
            embedding_function=embedding_func(),
            persist_directory=chroma_db_dir,
        )
        logger.info(f"[Stage 01, Part 06.1] Loading existing DB from path: {chroma_db_dir}.....")
        
        # calculate "page:chunk" IDs
        # Step 1: Assign chunk IDs
        chunks_with_ids = calc_chunk_ids(chunks, base_data_path, logger)
        logger.info(f"[Stage 01, Part 06.2] Calculated chunk IDs for total {len(chunks_with_ids)} chunks")

        # add/update the docs
        # Step 2: Get existing IDs from the DB
        existing_items = db.get(include=[]) # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        logger.info(f"[Stage 01, Part 06.2] No. of existing items (i.e. docs) in the db: {len(existing_ids)}")

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
        
        # Step 5: Filter out empty content chunks
        filtered_chunks = filter_chunks(unique_new_chunks, lower_limit, upper_limit, logger)
        if not filtered_chunks:
            logger.info("[Stage 01, Part 06.5] No valid (non-empty) chunks remaining after filtering")
            return
        
        # Step 6: Ingest in batches
        if unique_new_chunks:
            logger.info("[Stage 01, Part 07(a)] Ingesting new filtered chunks to DB in batches...")
            process_in_batches(
                documents=filtered_chunks,
                ingest_batch_size=ingest_batch_size,
                ingest_fn=lambda batch: db.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch]),
                logger=logger
            )
        else:
            logger.info("[Stage 01, Part 07(b)] No valid (non-empty) new unique chunks to add to DB")

        logger.info("[Stage 01, Part 08] Chunks saved to Chroma DB successfully")
    except Exception as e:
        logger.error(f"[Stage 01, Part 08] Error saving to Chroma DB: {e}")
        logger.debug(traceback.format_exc())
        return []

def clear_database(chroma_db_dir):
    if os.path.exists(chroma_db_dir):
        shutil.rmtree(chroma_db_dir)

def run_populate_db(reset=False, chroma_db_dir=CHROMA_DB_PATH, base_data_path=DATA_PATH, grouped_dirs=GROUPED_DIRS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, lower_limit=CHUNK_LOWER_LIMIT, upper_limit=CHUNK_UPPER_LIMIT, ingest_batch_size=INGEST_BATCH_SIZE):
    try:
        logger = setup_logger("populate_db_logger", LOG_FILE)
        logger.info(" ")
        logger.info("--------++++++++Starting db population stage.....")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("[Stage 01, Part 00.1] (RESET DB) Clearing the database...")
            clear_database(chroma_db_dir)
            logger.info("[Stage 01, Part 00.1] (RESET DB) Database cleared successfully.")
        
        # create (or update) the db
        docs = load_docs(base_data_path, grouped_dirs, logger)
        # logger.info(f"Loaded {len(docs)} docs")
        if not docs:
            logger.error("[Stage 01] No docs loaded. Exiting.")
            return
        
        flat_docs = flatten_docs(docs, logger)
        # logger.info(f"First document: {docs[0]}")
        
        chunks = split_docs(flat_docs, chunk_size, chunk_overlap, logger)
        # logger.info(f"Split into {len(chunks)} chunks")
        if not chunks:
            logger.error("[Stage 01] No chunks created. Exiting.")
            return
        # logger.info(f"First chunk: {chunks[0]}")
        
        # logger.info(f" [] Loaded {len(flat_docs)} docs, created {len(chunks)} chunks")
        save_to_chroma_db(chunks, chroma_db_dir, base_data_path, lower_limit, upper_limit, ingest_batch_size, logger)
        
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