import os
import gc
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Configuration
DATA_DIR = "C:/rag_for_query_resolution/data/collected_data"
CHROMA_DIR = "C:/rag_for_query_resolution/chromadb"
PROCESSED_LOG = os.path.join(CHROMA_DIR, "processed_files.txt")
OLLAMA_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 20 # decreased for reduction in similar embeddings
BATCH_SIZE = 30  # memory kam hai bhai

# Initialize shared components
embedding = OllamaEmbeddings(model=OLLAMA_MODEL)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    add_start_index=True,
)
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=CHROMA_DIR
)

# Deduplication helpers
def is_file_processed(file):
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r") as f:
            return file in f.read().splitlines()
    return False

def mark_file_processed(file):
    with open(PROCESSED_LOG, "a") as f:
        f.write(file + "\n")

# Generator to yield document batches
def process_docs_in_batches(docs, batch_size=BATCH_SIZE):
    for i in range(0, len(docs), batch_size):
        yield docs[i:i + batch_size]

# Load a single file
def load_documents_from_file(folder_path, file):
    file_path = os.path.join(folder_path, file)
    docs = []

    if file.endswith(".pdf"):
        try:
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
            print(f"✅ Loaded PDF: {file}")
        except Exception as e:
            print(f"❌ Skipped PDF {file}: {e}")

    elif file.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            for sheet_name, sheet_df in df.items():
                for idx, row in sheet_df.iterrows():
                    content = row.astype(str).dropna().str.cat(sep=" | ")
                    if content.strip():
                        docs.append(Document(
                            page_content=content,
                            metadata={"source": file, "sheet": sheet_name, "row": idx}
                        ))
            print(f"✅ Loaded Excel: {file}")
        except Exception as e:
            print(f"❌ Skipped Excel {file}: {e}")
    
    return docs

# Validate embeddings (e.g., not empty or all-zero)
def has_valid_embedding(docs):
    try:
        vectors = embedding.embed_documents([doc.page_content for doc in docs])
        valid_docs = []
        for doc, vector in zip(docs, vectors):
            if vector and not np.allclose(vector, 0):
                valid_docs.append(doc)
        return valid_docs
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return []

# Process a single folder
def process_folder(folder_name):
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder does not exist: {folder_path}")
        return

    for file in os.listdir(folder_path):
        if is_file_processed(file):
            print(f"⏩ Skipping already-processed file: {file}")
            continue

        all_docs = load_documents_from_file(folder_path, file)
        for doc_batch in process_docs_in_batches(all_docs):
            chunks = splitter.split_documents(doc_batch)
            valid_chunks = has_valid_embedding(chunks)
            if valid_chunks:
                try:
                    vectordb.add_documents(valid_chunks)
                    print(f"🧱 Stored {len(valid_chunks)} valid chunks from {file}")
                except ValueError as e:
                    print(f"❌ Skipped chunk batch from {file} due to embedding error: {e}")
            else:
                print(f"⚠️  No valid chunks with usable embeddings from {file}, skipping.")
            del doc_batch, chunks, valid_chunks
            gc.collect()

        mark_file_processed(file)

# Main ingestion pipeline
def main():
    print("🚀 Starting memory-efficient document processing...")
    for folder in ["core"]:
        print(f"📁 Processing folder: {folder}")
        process_folder(folder)
    print("✅ All documents processed and stored.")

# Interactive query loop
def query_loop():
    print("🧠 Initializing vector store for querying...")
    embedding = OllamaEmbeddings(model=OLLAMA_MODEL)
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )

    while True:
        query = input("🔍 Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        results = vectordb.similarity_search(query, k=10) #fetching more 
        seen = set()
        unique_results = []
        for doc in results:
            summary = doc.page_content.strip()[:300]
            if summary not in seen:
                seen.add(summary)
                unique_results.append(doc)
            if len(unique_results) >= 3:
                break
            
        print("\n📄 Top Results:\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. {doc.page_content[:300]}...\n")

# Entry point
if __name__ == "__main__":
    main()
    query_loop()
