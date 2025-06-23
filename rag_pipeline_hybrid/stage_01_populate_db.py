import os, shutil, gc, traceback, pickle
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.get_llm_func import embedding_func
from utils.hybrid_search import HybridRetriever
from utils.get_population_func import (
    load_docs, flatten_docs, split_docs, calc_chunk_ids, load_existing_chunks_metadata, filter_chunks, process_in_batches, save_processed_chunks_metadata
)
from langchain.schema import Document
from langchain_community.vectorstores import FAISS


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
DATA_PATH = Path(CONFIG["DATA_PATH"])
GROUPED_DIRS = CONFIG["GROUPED_DIRS"]

CHUNKS_OUT_PATH_HYBRID = Path(CONFIG["CHUNKS_OUT_PATH_HYBRID"])
HYBRID_FAISS_DB_DIR = CONFIG["HYBRID_FAISS_DB_DIR"]
HYBRID_BM25_DB_PATH = Path(CONFIG["HYBRID_BM25_DB_PATH"])

CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_LOWER_LIMIT = CONFIG["CHUNK_LOWER_LIMIT"]
CHUNK_UPPER_LIMIT = CONFIG["CHUNK_UPPER_LIMIT"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
INGEST_BATCH_SIZE = CONFIG["INGEST_BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "hybrid_stage_01_populate_db.log")


def save_to_faiss_db(chunks: list[Document], faiss_db_dir, bm25_db_path, base_data_path, chunks_dir, lower_limit, upper_limit, ingest_batch_size, logger):
    try:
        logger.info("[Stage 01, Part 06] Saving chunks to FAISS DB.....")

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
        
        # === FAISS DATABASE SECTION ===
        logger.info("[Stage 01, Part 07] Processing FAISS Database...")
        
        # Step 6: Load or create FAISS index
        faiss_db_dir = Path(faiss_db_dir)
        faiss_db_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = faiss_db_dir / "index.faiss"
        metadata_path = faiss_db_dir / "index.pkl"
        
        faiss_db = None
        if index_path.exists() and metadata_path.exists():
            # Load existing FAISS database
            logger.info(f"[Stage 01, Part 07.1] Loading existing FAISS DB from path: {faiss_db_dir}...")
            try:
                faiss_db = FAISS.load_local(
                    folder_path=str(faiss_db_dir),
                    embeddings=embedding_func(),
                    allow_dangerous_deserialization=True  # Required for loading pickled metadata
                )
                logger.info("[Stage 01, Part 07.1] Successfully loaded existing FAISS database")
            except Exception as e:
                logger.warning(f"[Stage 01, Part 07.1] Failed to load existing FAISS DB: {e}")
                logger.info("[Stage 01, Part 07.2] Creating new FAISS database...")
                faiss_db = None
        else:
            logger.info("[Stage 01, Part 07] No existing FAISS DB found. Will create new one.")
            faiss_db = None
        
        # Step 7: Process chunks in batches
        logger.info("[Stage 01, Part 08] Processing chunks in batches...")
        
        def add_batch_to_faiss(batch):
            nonlocal faiss_db
            try:
                if faiss_db is None:
                    # Create new FAISS database with first batch
                    logger.info(f"[Stage 01, Part 8.1] Creating new FAISS DB with batch of {len(batch)} documents")
                    faiss_db = FAISS.from_documents(
                        documents=batch,
                        embedding=embedding_func()
                    )
                else:
                    # Add to existing database
                    logger.info(f"[Stage 01, Part 08.2] Adding batch of {len(batch)} documents to existing FAISS DB")
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    faiss_db.add_texts(texts=texts, metadatas=metadatas)
                
            except Exception as e:
                logger.error(f"[Stage 01, Part 08] Error adding batch to FAISS: {e}")
                logger.debug(traceback.format_exc())
                raise e
        
        # Process in batches
        process_in_batches(
            documents=filtered_chunks,
            ingest_batch_size=ingest_batch_size,
            ingest_fn=add_batch_to_faiss,
            logger=logger
        )
        
        # Step 8: Save the updated FAISS database
        if faiss_db is not None:
            logger.info(f"[Stage 01, Part 09.1] Saving FAISS database to: {faiss_db_dir}...")
            faiss_db.save_local(folder_path=str(faiss_db_dir))
            logger.info("[Stage 01, Part 09.2] FAISS database saved successfully")
            
            # Step 9: Only after successful FAISS save, update the metadata tracking file
            logger.info("[Stage 01, Part 10] Updating processed chunks metadata file...")
            save_processed_chunks_metadata(filtered_chunks, chunks_dir, logger)
            logger.info("[Stage 01, Part 10] Chunks saved to FAISS DB successfully")
        else:
            logger.warning("[Stage 01, Part 10] No FAISS database to save")
            return
        
        # === BM25 DATABASE SECTION ===
        logger.info("[Stage 01, Part 09] Processing BM25 Database...")
        
        try:
            # Step 9: Load existing BM25 index if it exists
            hybrid_retriever = HybridRetriever()
            existing_documents = []
            
            bm25_db_path = Path(bm25_db_path)
            bm25_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            if bm25_db_path.exists():
                logger.info(f"[Stage 01, Part 09.1] Loading existing BM25 index from: {bm25_db_path}")
                try:
                    with open(bm25_db_path, "rb") as f:
                        saved_data = pickle.load(f)
                        if isinstance(saved_data, dict) and 'documents' in saved_data:
                            existing_documents = saved_data['documents']
                            logger.info(f"[Stage 01, Part 09.1] Loaded {len(existing_documents)} existing documents from BM25 index")
                        else:
                            logger.warning("[Stage 01, Part 09.1] Invalid BM25 index format, creating new one")
                except Exception as e:
                    logger.warning(f"[Stage 01, Part 09.1] Failed to load existing BM25 index: {e}")
            else:
                logger.info("[Stage 01, Part 09.1] No existing BM25 index found, creating new one")
            
            # Step 10: Combine existing documents with new ones
            all_documents = existing_documents + filtered_chunks
            logger.info(f"[Stage 01, Part 09.2] Total documents for BM25 indexing: {len(all_documents)}")
            
            # Step 11: Build BM25 index with all documents
            logger.info("[Stage 01, Part 09.3] Building BM25 index...")
            hybrid_retriever.build_bm25_index(all_documents)
            
            # Step 12: Save BM25 index
            logger.info(f"[Stage 01, Part 09.4] Saving BM25 index to: {bm25_db_path}")
            bm25_data = {
                'bm25_index': hybrid_retriever.bm25_index,
                'tokenized_corpus': hybrid_retriever.tokenized_corpus,
                'documents': all_documents
            }
            
            with open(bm25_db_path, "wb") as f:
                pickle.dump(bm25_data, f)
            logger.info("[Stage 01, Part 09.4] BM25 index saved successfully")
            
        except Exception as e:
            logger.error(f"[Stage 01, Part 09] Error creating BM25 index: {e}")
            logger.debug(traceback.format_exc())
            # Don't return here - FAISS was successful, so we can continue

        # Step 13: Update metadata tracking file (only after both databases are processed)
        logger.info("[Stage 01, Part 10] Updating processed chunks metadata file...")
        save_processed_chunks_metadata(filtered_chunks, chunks_dir, logger)

        logger.info("[Stage 01, Part 10] Chunks saved to Hybrid DB (FAISS + BM25) successfully")
        
    except Exception as e:
        logger.error(f"[Stage 01, Part 10] Error saving to Hybrid DB: {e}")
        logger.debug(traceback.format_exc())
        return []

def clear_database(faiss_db_dir, bm25_db_path):
    if os.path.exists(faiss_db_dir):
        shutil.rmtree(faiss_db_dir)
    
    bm25_path = Path(bm25_db_path)
    if bm25_path.exists():
        bm25_path.unlink()

def clear_chunks(chunks_dir):
    chunks_path = Path(chunks_dir)
    if chunks_path.exists():
        if chunks_path.is_file():
            chunks_path.unlink()
        elif chunks_path.is_dir():
            shutil.rmtree(chunks_path)

def run_populate_db(reset=False, faiss_db_dir=HYBRID_FAISS_DB_DIR, bm25_db_path=HYBRID_BM25_DB_PATH, base_data_path=DATA_PATH, chunks_dir=CHUNKS_OUT_PATH_HYBRID, grouped_dirs=GROUPED_DIRS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, lower_limit=CHUNK_LOWER_LIMIT, upper_limit=CHUNK_UPPER_LIMIT, ingest_batch_size=INGEST_BATCH_SIZE):
    try:
        logger = setup_logger("populate_hybrid_db_logger", LOG_FILE)
        logger.info(" ")
        logger.info("--------++++++++Starting Hybrid Db population stage.....")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("[Stage 01, Part 00.1] (RESET DB) Clearing the databases...")
            clear_database(faiss_db_dir)
            logger.info("[Stage 01, Part 00.1] (RESET DB) Databases cleared successfully.")
            
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
        
        save_to_faiss_db(chunks, faiss_db_dir, bm25_db_path, base_data_path, chunks_dir, lower_limit, upper_limit, ingest_batch_size, logger)
        
        # Manual memory cleanup
        del docs, flat_docs, chunks
        gc.collect()
        
        logger.info("--------++++++++Hybrid Db population stage successfully completed.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error at [Stage 01]: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_populate_db()