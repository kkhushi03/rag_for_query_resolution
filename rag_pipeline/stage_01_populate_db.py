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
from langchain_chroma import Chroma


# configurations
LOG_PATH = CONFIG["LOG_PATH"]
LOG_PATH = Path(LOG_PATH)
DATA_PATH = CONFIG["DATA_PATH"]
DATA_PATH = Path(DATA_PATH)
GROUPED_DIRS = CONFIG["GROUPED_DIRS"]

CHUNKS_OUT_PATH = CONFIG["CHUNKS_OUT_PATH"]
CHUNKS_OUT_PATH = Path(CHUNKS_OUT_PATH)
EMBEDDINGS_OUT_PATH = CONFIG["EMBEDDINGS_OUT_PATH"]
EMBEDDINGS_OUT_PATH = Path(EMBEDDINGS_OUT_PATH)
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


# def load_docs(base_data_path, logger) -> List[Document]:
#     try:
#         logger.info("[Stage 01, Part 01.1] Loading docs from the directory.....")
        
#         all_docs = []
#         pdf_files = []
#         logger.info(f"[Stage 01, Part 01.2] Scanning directory.....: {base_data_path}")
        
#         try:
#             # Collect all PDF paths (including subfolders)
#             # Loop 1: directory walk
#             for root, _, files in os.walk(base_data_path):
#                 # Loop 2: files in that folder
#                 for file in files:
#                     if file.lower().endswith(".pdf"):
#                         root_file_path = os.path.join(root, file)
#                         pdf_files.append(root_file_path)
#                         logger.debug(f"Found PDF file: {root_file_path}")
            
#             logger.info(f"[Stage 01, Part 02] Loading.....: {len(pdf_files)}")
            
#             # Loop 3: process all PDFs
#             for file_path in tqdm(pdf_files, desc="Loading PDFs"):
#                 try:
#                     loader = PyMuPDFLoader(file_path)
#                     docs = loader.load()
#                     all_docs.extend(docs)
#                 except Exception as inner_e:
#                     logger.warning(f"Failed to load {file_path}: {inner_e}")
#                     logger.debug(traceback.format_exc())
#                     continue
#         finally:
#             gc.collect()  # Free memory after each file
        
#         logger.info(f"[Stage 01, Part 03] Loaded {len(all_docs)} documents successfully.")
#         return all_docs
#     except Exception as e:
#         logger.error(f"[Stage 01, Part 03] Error loading documents: {e}")
#         logger.debug(traceback.format_exc())
#         return []

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

def serialize_chunks(chunks, chunks_dir, logger):
    logger.info(f"[Stage 01, Part 05.3] Serializing {len(chunks)} chunks to {chunks_dir}...")
    
    try:
        chunks_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_dir, "w", encoding="utf-8") as f:
            for chunk in chunks:
                logger.debug(f"[Stage 01, Part 05.3] Serializing chunk: {chunk}.....")
                f.write(json.dumps({"content": chunk.page_content, "metadata": chunk.metadata}) + "\n")
                logger.debug(f"[Stage 01, Part 05.3] Serialized chunk: {chunk}")
                
        logger.info(f"[Stage 01, Part 05.4] Serialized {len(chunks)} chunks to {chunks_dir} successfully.")
    except Exception as e:
        logger.error(f"[Stage 01, Part 05] Error serializing chunks: {e}")
        logger.debug(traceback.format_exc())

def calc_chunk_ids(chunks, base_data_path, logger):
    try:
        # Page Source : Page Number : Chunk Index
        last_page_id = None
        curr_chunk_idx = 0
        
        logger.info("[Stage 01, Part 07.1] Calculating chunk IDs...")
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
        
        logger.info("[Stage 01, Part 07.2] Chunk IDs calculated successfully.")
        return chunks
    except Exception as e:
        logger.error(f"[Stage 01, Part 07] Error calculating chunk IDs: {e}")
        logger.debug(traceback.format_exc())
        return chunks

def filter_chunks(chunks: List[Document], lower_limit: int, upper_limit: int, logger) -> List[Document]:
    logger.info("[Stage 01, Part 08.4.1] Applying custom (token-based) filtering on chunks before ingestion...")
    
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
        
        logger.info(f"[Stage 01, Part 08.4.2(a)] Chunks remaining before filter: {len(chunks)}")
        logger.info(f"[Stage 01, Part 08.4.2(b)] Chunks remaining after filter: {len(filtered)}")
        return filtered
    except Exception as e:
        logger.error(f"[Stage 01, Part 08.4] Error applying custom filtering on chunks: {e}")
        logger.debug(traceback.format_exc())
        return chunks

def process_in_batches(documents, ingest_batch_size: int, ingest_fn, logger):
    total = len(documents)
    logger.info(f"[Stage 01, Part 09.1] Processing {total} documents in batches of {ingest_batch_size}...")

    for i in tqdm(range(0, total, ingest_batch_size), desc="Ingesting batches"):
        batch = documents[i:i + ingest_batch_size]
        try:
            ingest_fn(batch)
            logger.info(f"[Stage 01, Part 09.2] Ingested batch {i // ingest_batch_size}")
        except Exception as e:
            logger.error(f"[Stage 01, Part 09] Failed to ingest batch {i // ingest_batch_size}: {e}")
            logger.debug(traceback.format_exc())

def save_to_chroma_db(chunks: list[Document], chroma_db_dir, base_data_path, lower_limit, upper_limit, ingest_batch_size, logger):
    try:
        logger.info("[Stage 01, Part 06] Saving chunks to Chroma DB.....")
        
        # load the existing db
        db = Chroma(
            embedding_function=embedding_func(),
            persist_directory=chroma_db_dir,
        )
        logger.info(f"[Stage 01, Part 07] Loading existing DB from path: {chroma_db_dir}.....")
        
        # calculate "page:chunk" IDs
        # Step 1: Assign chunk IDs
        chunks_with_ids = calc_chunk_ids(chunks, base_data_path, logger)
        logger.info(f"[Stage 01, Part 08.1] Calculated chunk IDs for total {len(chunks_with_ids)} chunks")
        
        # add/update the docs
        # Step 2: Get existing IDs from the DB
        existing_items = db.get(include=[]) # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        logger.info(f"[Stage 01, Part 08.2] No. of existing items (i.e. docs) in the db: {len(existing_ids)}")
        
        # only add docs that don't exist in the db
        # Step 3: Filter only new chunks
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["chunk_id"] not in existing_ids:
                new_chunks.append(chunk)
        logger.info(f"[Stage 01, Part 08.3] No. of new chunks to add: {len(new_chunks)}")
        
        # Step 4: Safety Net — remove duplicates within `new_chunks` -> Deduplicate
        seen_ids = set()
        unique_new_chunks = []
        for chunk in new_chunks:
            cid = chunk.metadata["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_new_chunks.append(chunk)
        logger.info(f"[Stage 01, Part 08.4] No. of unique new chunks to add, after deduplication: {len(unique_new_chunks)}")
        
        # Step 5: Filter out empty content chunks
        filtered_chunks = filter_chunks(unique_new_chunks, lower_limit, upper_limit, logger)
        
        # Step 6: Ingest in batches
        if unique_new_chunks:
            logger.info("[Stage 01, Part 09(a)] Ingesting new filtered chunks to DB in batches...")
            process_in_batches(
                documents=filtered_chunks,
                ingest_batch_size=ingest_batch_size,
                ingest_fn=lambda batch: db.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch]),
                logger=logger
            )
        else:
            logger.info("[Stage 01, Part 09(b)] No valid (non-empty) new unique chunks to add to DB")
        
        logger.info("[Stage 01, Part 10] Chunks saved to Chroma DB successfully")        
    except Exception as e:
        logger.error(f"[Stage 01, Part 10] Error saving to Chroma DB: {e}")
        logger.debug(traceback.format_exc())
        return []

def clear_database(chroma_db_dir):
    if os.path.exists(chroma_db_dir):
        shutil.rmtree(chroma_db_dir)

def run_populate_db(reset=False, chroma_db_dir=CHROMA_DB_PATH, base_data_path=DATA_PATH, chunks_dir=CHUNKS_OUT_PATH, grouped_dirs=GROUPED_DIRS, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, lower_limit=CHUNK_LOWER_LIMIT, upper_limit=CHUNK_UPPER_LIMIT, ingest_batch_size=INGEST_BATCH_SIZE):
    try:
        logger = setup_logger("populate_db_logger", LOG_FILE)
        logger.info(" ")
        logger.info("--------++++++++Starting db population stage.....")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("[Stage 01, Part 00] (RESET DB) Clearing the database...")
            clear_database(chroma_db_dir)
        
        # create (or update) the db
        # docs = load_docs(base_data_path, logger)
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
        serialized_chunks = serialize_chunks(chunks, chunks_dir, logger)
        
        save_to_chroma_db(chunks, chroma_db_dir, base_data_path, lower_limit, upper_limit, ingest_batch_size, logger)
        
        # Manual memory cleanup
        del docs, flat_docs, chunks, serialized_chunks
        gc.collect()
        
        logger.info("--------++++++++DB population stage successfully completed.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error at [Stage 01]: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_populate_db()