import os, json, shutil, gc, traceback
from tqdm import tqdm
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.get_llm_func import embedding_func, num_tokens
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
DATA_PATH = Path(CONFIG["DATA_PATH"])
GROUPED_DIRS = CONFIG["GROUPED_DIRS"]

CHUNKS_OUT_PATH_FAISS = Path(CONFIG["CHUNKS_OUT_PATH_FAISS"])
FAISS_DB_PATH = CONFIG["FAISS_DB_PATH"]

CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_LOWER_LIMIT = CONFIG["CHUNK_LOWER_LIMIT"]
CHUNK_UPPER_LIMIT = CONFIG["CHUNK_UPPER_LIMIT"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
INGEST_BATCH_SIZE = CONFIG["INGEST_BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db.log")


def load_docs(base_data_path: Path, grouped_dirs: List[str], logger) -> List[Document]:
    try:
        logger.info("[Stage 01, Part 01.1] Loading docs from selected grouped directories.....")
        all_docs = []
        total_pdfs = 0

        for group_dir in grouped_dirs:
            full_group_path = base_data_path / group_dir
            logger.info(f"[Stage 01, Part 01.2] Scanning directory.....: {full_group_path}")

            if not full_group_path.exists():
                logger.warning(f"[Stage 01, Part 01.2] Directory does not exist: {full_group_path}")
                continue

            pdf_files = list(full_group_path.rglob("*.pdf"))
            total_pdfs += len(pdf_files)
            logger.info(f"[Stage 01, Part 02] Found {len(pdf_files)} PDFs in {group_dir}")
            
            logger.info(f"[Stage 01, Part 02] Loading.....: {len(pdf_files)}")
            for pdf_file in tqdm(pdf_files, desc=f"Loading PDFs from {group_dir}"):
                try:
                    loader = PyMuPDFLoader(str(pdf_file))
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as inner_e:
                    logger.warning(f"[Part 02] Failed to load {pdf_file}: {inner_e}")
                    logger.debug(traceback.format_exc())
                    continue
                finally:
                    gc.collect() # Free memory after each file

        logger.info(f"[Stage 01, Part 03.1] Total PDFs processed: {total_pdfs}")
        logger.info(f"[Stage 01, Part 03.2] Total documents loaded: {len(all_docs)}")
        return all_docs

    except Exception as e:
        logger.error(f"[Stage 01, Part 03] Error during document loading: {e}")
        logger.debug(traceback.format_exc())
        return []

def flatten_docs(docs: List, logger) -> List[Document]:
    try:
        logger.info("[Stage 01, Part 04] Flattening docs list.....")
        if docs and isinstance(docs[0], list):
            docs_list = [item for sublist in docs for item in sublist]
            logger.info(f"[Stage 01, Part 04(a)] Flattened docs count: {len(docs_list)}")
        else:
            docs_list = docs
            logger.info(f"[Stage 01, Part 04(b)] Docs already flat, count: {len(docs_list)}")
    except Exception as e:
        logger.error(f"[Stage 01, Part 04] Error flattening docs list: {e}")
        logger.debug(traceback.format_exc())
        docs_list = docs  # fallback
    return docs_list

def split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int, logger) -> List[Document]:
    try:
        logger.info("[Stage 01, Part 05] Splitting docs into chunks....")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # length_function=len,
            # is_separator_regex=False,
        )
        logger.info(f"[Stage 01, Part 05.1] Splitting {len(docs)} documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
        chunks = text_splitter.split_documents(docs)
        
        logger.info(f"[Stage 01, Part 05.2] Total chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"[Stage 01, Part 05] Error splitting documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def calc_chunk_ids(chunks, base_data_path, chunks_dir, logger):
    try:
        # Page Source : Page Number : Chunk Index
        last_page_id = None
        curr_chunk_idx = 0
        
        logger.info("[Stage 01, Part 06.1.1] Calculating chunk IDs...")
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            
            # logger.info("[Stage 01, Part 11.1] Normalizing & Standardizing paths, for cross-platform consistency...")
            norm_source = os.path.normpath(source)
            rel_source = os.path.relpath(norm_source, base_data_path).replace("\\", "/")
            
            curr_page_id = (f"{rel_source}:{page}")
            
            # if the page ID is the same as the last one, increment the index
            if curr_page_id == last_page_id:
                curr_chunk_idx += 1
            else:
                curr_chunk_idx = 0
            
            # logger.info("[Stage 01, Part 11.2] Calculating new chunk IDs...")
            chunk_id = (f"{curr_page_id}:{curr_chunk_idx}")
            last_page_id = curr_page_id
            
            # logger.info("[Stage 01, Part 11.3] Adding chunk IDs to metadata...")
            chunk.metadata["chunk_id"] = chunk_id
        
        logger.info("[Stage 01, Part 06.1.2] Chunk IDs calculated successfully.")
        return chunks
    except Exception as e:
        logger.error(f"[Stage 01, Part 06.1] Error calculating chunk IDs: {e}")
        logger.debug(traceback.format_exc())
        return chunks

def load_existing_chunks_metadata(chunks_dir, logger):
    try:
        if not os.path.exists(chunks_dir):
            logger.warning(f"[Stage 01, Part 06.2] No existing chunks file found at {chunks_dir}")
            return set()
        
        existing_ids = set()
        with open(chunks_dir, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk_data = json.loads(line.strip())
                    chunk_id = chunk_data.get("metadata", {}).get("chunk_id")
                    if chunk_id:
                        existing_ids.add(chunk_id)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"[Stage 01, Part 06.2] Loaded {len(existing_ids)} existing chunk IDs")
        return existing_ids
    except Exception as e:
        logger.error(f"[Stage 01, Part 06.2] Error loading existing chunks metadata: {e}")
        logger.debug(traceback.format_exc())
        return set()

def filter_chunks(chunks: List[Document], lower_limit: int, upper_limit: int, logger) -> List[Document]:
    logger.info("[Stage 01, Part 06.5.1] Applying custom (token-based) filtering on chunks before ingestion...")
    
    try:
        filtered = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            token_len = num_tokens(content)
            
            # only chunks between lower_limit and upper_limit characters
            if lower_limit < token_len < upper_limit:
                filtered.append(chunk)
            # else:
            #     logger.warning(f"[Stage 01, Part 14.4.2(a)] Filtered out chunk ((tokens={token_len}, len={len(content)}): {chunk.metadata.get('chunk_id')}")
        
        logger.info(f"[Stage 01, Part 06.5.2(a)] Chunks remaining before filter: {len(chunks)}")
        logger.info(f"[Stage 01, Part 06.5.2(b)] Chunks remaining after filter: {len(filtered)}")
        return filtered
    except Exception as e:
        logger.error(f"[Stage 01, Part 06.5] Error applying custom filtering on chunks: {e}")
        logger.debug(traceback.format_exc())
        return chunks

def process_in_batches(documents, ingest_batch_size: int, ingest_fn, logger):
    total = len(documents)
    logger.info(f"[Stage 01, Part 08.3] Processing {total} documents in batches of {ingest_batch_size}...")

    for i in tqdm(range(0, total, ingest_batch_size), desc="Ingesting batches"):
        batch = documents[i:i + ingest_batch_size]
        try:
            ingest_fn(batch)
            logger.info(f"[Stage 01, Part 08.4] Ingested batch {i // ingest_batch_size}")
        except Exception as e:
            logger.error(f"[Stage 01, Part 08.4] Failed to ingest batch {i // ingest_batch_size}: {e}")
            logger.debug(traceback.format_exc())

def save_processed_chunks_metadata(chunks, chunks_dir, logger):
    # Save successfully processed chunks metadata to JSON file for tracking
    try:
        logger.info(f"[Stage 01, Part 10.1.1] Saving processed chunks metadata to: {chunks_dir}")
        chunks_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Read existing data first
        existing_data = []
        if os.path.exists(chunks_dir):
            with open(chunks_dir, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        existing_data.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        # Append new chunks
        with open(chunks_dir, "a", encoding="utf-8") as f:
            for chunk in chunks:
                chunk_data = {
                    "content": chunk.page_content, 
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(chunk_data) + "\n")
        
        logger.info(f"[Stage 01, Part 10.1.2] Saved {len(chunks)} processed chunks to metadata file")
        
    except Exception as e:
        logger.error(f"[Stage 01, Part 10.1] Error saving processed chunks metadata: {e}")
        logger.debug(traceback.format_exc())
    
    # Load existing chunks metadata from JSON file to track what's already processed
    try:
        if not os.path.exists(chunks_dir):
            logger.info(f"[Stage 01, Part 10.2.1] No existing chunks file found at {chunks_dir}")
            return set()
        
        existing_ids = set()
        with open(chunks_dir, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk_data = json.loads(line.strip())
                    chunk_id = chunk_data.get("metadata", {}).get("chunk_id")
                    if chunk_id:
                        existing_ids.add(chunk_id)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"[Stage 01, Part 10.2.2] Loaded {len(existing_ids)} existing chunk IDs")
        return existing_ids
    except Exception as e:
        logger.error(f"[Stage 01, Part 10.2] Error loading existing chunks metadata: {e}")
        logger.debug(traceback.format_exc())
        return set()

def save_to_faiss_db(chunks: list[Document], faiss_db_dir, base_data_path, chunks_dir, lower_limit, upper_limit, ingest_batch_size, logger):
    try:
        logger.info("[Stage 01, Part 06] Saving chunks to FAISS DB.....")

        # calculate "page:chunk" IDs
        # Step 1: Assign chunk IDs
        chunks_with_ids = calc_chunk_ids(chunks, base_data_path, chunks_dir, logger)
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
        
        # Step 6: Load or create FAISS index
        faiss_db_dir = Path(faiss_db_dir)
        faiss_db_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = faiss_db_dir / "index.faiss"
        metadata_path = faiss_db_dir / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            # Load existing FAISS database
            logger.info(f"[Stage 01, Part 07] Loading existing FAISS DB from path: {faiss_db_dir}...")
            try:
                db = FAISS.load_local(
                    folder_path=str(faiss_db_dir),
                    embeddings=embedding_func(),
                    allow_dangerous_deserialization=True  # Required for loading pickled metadata
                )
                logger.info("[Stage 01, Part 07.1] Successfully loaded existing FAISS database")
            except Exception as e:
                logger.warning(f"[Stage 01, Part 07.1] Failed to load existing FAISS DB: {e}")
                logger.info("[Stage 01, Part 07.2] Creating new FAISS database...")
                db = None
        else:
            logger.info("[Stage 01, Part 07] No existing FAISS DB found. Will create new one.")
            db = None
        
        # Step 7: Process chunks in batches
        logger.info("[Stage 01, Part 08] Processing chunks in batches...")
        
        def add_batch_to_faiss(batch):
            nonlocal db
            try:
                if db is None:
                    # Create new FAISS database with first batch
                    logger.info(f"[Stage 01, Part 8.1] Creating new FAISS DB with batch of {len(batch)} documents")
                    db = FAISS.from_documents(
                        documents=batch,
                        embedding=embedding_func()
                    )
                else:
                    # Add to existing database
                    logger.info(f"[Stage 01, Part 08.2] Adding batch of {len(batch)} documents to existing FAISS DB")
                    texts = [doc.page_content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    db.add_texts(texts=texts, metadatas=metadatas)
                
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
        if db is not None:
            logger.info(f"[Stage 01, Part 09.1] Saving FAISS database to: {faiss_db_dir}...")
            db.save_local(folder_path=str(faiss_db_dir))
            logger.info("[Stage 01, Part 09.2] FAISS database saved successfully")
            
            # Step 9: Only after successful FAISS save, update the metadata tracking file
            logger.info("[Stage 01, Part 10] Updating processed chunks metadata file...")
            save_processed_chunks_metadata(filtered_chunks, chunks_dir, logger)
        else:
            logger.warning("[Stage 01, Part 10] No FAISS database to save")

        logger.info("[Stage 01, Part 10] Chunks saved to FAISS DB successfully")
    except Exception as e:
        logger.error(f"[Stage 01, Part 10] Error saving to FAISS DB: {e}")
        logger.debug(traceback.format_exc())
        return []

def clear_database(faiss_db_dir):
    if os.path.exists(faiss_db_dir):
        shutil.rmtree(faiss_db_dir)

def clear_chunks(chunks_dir):
    chunks_path = Path(chunks_dir)
    if chunks_path.exists():
        if chunks_path.is_file():
            chunks_path.unlink()
        elif chunks_path.is_dir():
            shutil.rmtree(chunks_path)

def run_populate_db(reset=False, faiss_db_dir=FAISS_DB_PATH, base_data_path=DATA_PATH, chunks_dir=CHUNKS_OUT_PATH_FAISS, grouped_dirs=GROUPED_DIRS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, lower_limit=CHUNK_LOWER_LIMIT, upper_limit=CHUNK_UPPER_LIMIT, ingest_batch_size=INGEST_BATCH_SIZE):
    try:
        logger = setup_logger("populate_db_logger", LOG_FILE)
        logger.info(" ")
        logger.info("--------++++++++Starting db population stage.....")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("[Stage 01, Part 00.1] (RESET DB) Clearing the database...")
            clear_database(faiss_db_dir)
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
        
        save_to_faiss_db(chunks, faiss_db_dir, base_data_path, chunks_dir, lower_limit, upper_limit, ingest_batch_size, logger)
        
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