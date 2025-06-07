import os
import gc
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# Configuration
DATA_DIR = "C:/rag_for_query_resolution/data/collected_data"
CHROMA_DIR = "C:/rag_for_query_resolution/chromadb"
PROCESSED_LOG = os.path.join(CHROMA_DIR, "processed_files.txt")
OLLAMA_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 20
BATCH_SIZE = 30

# Shared Components
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

# Deduplication
def is_file_processed(file):
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, "r") as f:
            return file in f.read().splitlines()
    return False

def mark_file_processed(file):
    with open(PROCESSED_LOG, "a") as f:
        f.write(file + "\n")

# Batching Generator
def process_docs_in_batches(docs, batch_size=BATCH_SIZE):
    for i in range(0, len(docs), batch_size):
        yield docs[i:i + batch_size]

# Load Documents
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

# Validate Embeddings
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

# Grading Setup
llm = ChatOllama(model=LLM_MODEL)
grading_prompt = PromptTemplate.from_template("""
You are an expert assistant evaluating the relevance of a context passage to a user query.

Query:
"{query}"

Context:
"{context}"

On a scale of 0 to 10:
- 10 = Extremely relevant and directly answers the query
- 7–9 = Mostly relevant, with strong information alignment
- 4–6 = Somewhat relevant, may be partially useful
- 1–3 = Slightly relevant or tangential
- 0 = Not relevant at all

Return only the number from 0 to 10.
""")
grader_chain = grading_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: int(x.strip()))

# Folder Processing
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

# Main Pipeline
def main():
    print("🚀 Starting memory-efficient document processing...")
    for folder in ["core", "arxiv", "pdfs", "excels"]:
        print(f"📁 Processing folder: {folder}")
        process_folder(folder)
    print("✅ All documents processed and stored.")

# Querying with Grading
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

        results_with_scores = vectordb.similarity_search_with_score(query, k=10)

        graded_results = []
        for doc, _ in results_with_scores:
            try:
                score = grader_chain.invoke({"query": query, "context": doc.page_content})
                graded_results.append((doc, score))
            except Exception as e:
                print(f"⚠️  Grading failed: {e}")

        graded_results.sort(key=lambda x: -x[1])
        top_k = graded_results[:3]

        print("\n📄 Top Graded Results:\n")
        for i, (doc, grade) in enumerate(top_k, 1):
            source_file = doc.metadata.get("source", "Unknown Source")
            print(f"{i}. 📘 From: {os.path.basename(source_file)} | 🏷️ Grade: {grade}/10")
            print(f"{doc.page_content[:2000]}...\n")

# Entry Point
if __name__ == "__main__":
    main()
    query_loop()
