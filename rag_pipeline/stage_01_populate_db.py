import os
import yaml
import traceback
import shutil
import argparse
from tqdm import tqdm
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.get_llm_func import embedding_func
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


LOG_PATH = CONFIG["LOG_PATH"]
DATA_PATH = CONFIG["DATA_PATH"]
CHROMA_DB_PATH = CONFIG["CHROMA_DB_PATH"]
CHUNK_SIZE = CONFIG["CHUNK_SIZE"]
CHUNK_OVERLAP = CONFIG["CHUNK_OVERLAP"]
BATCH_SIZE = CONFIG["BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db.log")


def load_docs(data_dir, logger) -> List[Document]:
    try:
        logger.info("[Part 01] Loading docs from the directory.....")
        
        all_docs = []
        pdf_files = []
        logger.info(f"[Part 02] Scanning directory.....: {data_dir}")
        
        # Collect all PDF paths (including subfolders)
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
                    logger.info(f"[Part 03] Loading.....: {len(pdf_files)}")
                    
                    for file_path in tqdm(pdf_files, desc="Loading PDFs"):
                        try:
                            loader = PyMuPDFLoader(file_path)
                            docs = loader.load()
                            all_docs.extend(docs)
                        except Exception as inner_e:
                            logger.warning(f"Failed to load {file_path}: {inner_e}")
        
        logger.info(f"[Part 04] Loaded {len(all_docs)} documents successfully.")
        
        return all_docs
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def flatten_docs(docs: List, logger) -> List[Document]:
    try:
        logger.info("[Part 05] Flattening docs list.....")
        if docs and isinstance(docs[0], list):
            docs_list = [item for sublist in docs for item in sublist]
            logger.info(f"[Part 06(a)] Flattened docs count: {len(docs_list)}")
        else:
            docs_list = docs
            logger.info(f"[Part 06(b)] Docs already flat, count: {len(docs_list)}")
    except Exception as e:
        logger.error(f"Error flattening docs list: {e}")
        logger.debug(traceback.format_exc())
        docs_list = docs  # fallback
    return docs_list

def split_docs(docs: List[Document], chunk_size: int, chunk_overlap: int, logger) -> List[Document]:
    try:
        logger.info("[Part 07] Splitting docs into chunks....")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # length_function=len,
            # is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"[Part 08] Total chunks created: {len(chunks)}")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def process_in_batches(documents, batch_size, ingest_fn, logger):
    total = len(documents)
    logger.info(f"[Part 09] Processing {total} documents in batches of {batch_size}...")

    for i in tqdm(range(0, total, batch_size), desc="Ingesting batches"):
        batch = documents[i:i + batch_size]
        try:
            ingest_fn(batch)
            logger.info(f"[Part 10] Ingested batch {i // batch_size}")
        except Exception as e:
            logger.error(f"Failed to ingest batch {i // batch_size}: {e}")
            logger.debug(traceback.format_exc())

def calc_chunk_ids(chunks, data_dir, logger):
    try:
        # Page Source : Page Number : Chunk Index
        last_page_id = None
        curr_chunk_idx = 0
        
        logger.info("[Part 11] Calculating chunk IDs...")
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            
            logger.info("[Part 12] Normalizing & Standardizing paths, for cross-platform consistency...")
            norm_source = os.path.normpath(source)
            rel_source = os.path.relpath(norm_source, data_dir).replace("\\", "/")
            
            curr_page_id = (f"{rel_source}:{page}")
            
            # if the page ID is the same as the last one, increment the index
            if curr_page_id == last_page_id:
                curr_chunk_idx += 1
            else:
                curr_chunk_idx = 0
            
            logger.info("[Part 13] Calculating new chunk IDs...")
            chunk_id = (f"{curr_page_id}:{curr_chunk_idx}")
            last_page_id = curr_page_id
            
            logger.info("[Part 14] Adding chunk IDs to metadata...")
            chunk.metadata["chunk_id"] = chunk_id
        
        logger.info("[Part 15] Chunk IDs calculated successfully.")
        return chunks
    except Exception as e:
        logger.error(f"Error calculating chunk IDs: {e}")
        logger.debug(traceback.format_exc())
        return chunks

def save_to_chroma_db(chunks: list[Document], chroma_db_dir, data_dir, batch_size, logger):
    try:
        logger.info("[Part 16] Saving chunks to Chroma DB.....")
        
        # load the existing db
        db = Chroma(
            embedding_function=embedding_func(),
            persist_directory=chroma_db_dir,
        )
        logger.info(f"[Part 17.1] Loading existing DB from path: {chroma_db_dir}.....")
        
        # calculate "page:chunk" IDs
        # Step 1: Assign chunk IDs
        chunks_with_ids = calc_chunk_ids(chunks, data_dir)
        logger.info(f"[Part 17.2] Calculated chunk IDs for total {len(chunks_with_ids)} chunks")
        
        # add/update the docs
        # Step 2: Get existing IDs from the DB
        existing_items = db.get(include=[]) # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        logger.info(f"[Part 17.3] No. of existing items (i.e. docs) in the db: {len(existing_ids)}")
        
        # only add docs that don't exist in the db
        # Step 3: Filter only new chunks
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["chunk_id"] not in existing_ids:
                new_chunks.append(chunk)
        logger.info(f"[Part 17.4] No. of new chunks to add: {len(new_chunks)}")
        
        # Step 4: Safety Net — remove duplicates within `new_chunks`
        seen_ids = set()
        unique_new_chunks = []
        for chunk in new_chunks:
            cid = chunk.metadata["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_new_chunks.append(chunk)
        logger.info(f"[Part 17.5] No. of unique new chunks to add, after deduplication: {len(unique_new_chunks)}")
        
        # Step 5: Ingest in batches
        if unique_new_chunks:
            logger.info("[Part 17.6(a)] Ingesting new chunks to DB in batches...")
            process_in_batches(
                documents=unique_new_chunks,
                batch_size=batch_size,
                ingest_fn=lambda batch: db.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch]),
                logger=logger
            )
        else:
            logger.info("[Part 17.6(b)] No new unique chunks to add to DB")
        
        logger.info("[Part 18] Chunks saved to Chroma DB successfully")        
    except Exception as e:
        logger.error(f"Error saving to Chroma DB: {e}")
        logger.debug(traceback.format_exc())
        return []

def clear_database(chroma_db_dir):
    if os.path.exists(chroma_db_dir):
        shutil.rmtree(chroma_db_dir)

def run_populate_db(reset=False, chroma_db_dir=CHROMA_DB_PATH, data_dir=DATA_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, batch_size=BATCH_SIZE):
    try:
        logger = setup_logger("populate_db_logger", LOG_FILE)
        logger.info(" ")
        logger.info("--------++++++++Starting db population stage.....")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("[Part 19] (RESET DB) Clearing the database...")
            clear_database(chroma_db_dir)
        
        # create (or update) the db
        docs = load_docs(data_dir, logger)
        # logger.info(f"Loaded {len(docs)} docs")
        if not docs:
            logger.error("No docs loaded. Exiting.")
            return
        flat_docs = flatten_docs(docs, logger)
        # logger.info(f"First document: {docs[0]}")
        
        chunks = split_docs(flat_docs, chunk_size, chunk_overlap, logger)
        # logger.info(f"Split into {len(chunks)} chunks")
        if not chunks:
            logger.error("No chunks created. Exiting.")
            return
        # logger.info(f"First chunk: {chunks[0]}")
        
        save_to_chroma_db(chunks, chroma_db_dir, data_dir, batch_size, logger)
        logger.info("--------++++++++DB population stage successfully completed.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_populate_db()