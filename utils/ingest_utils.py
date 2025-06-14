
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_documents_from_dirs(base_dir, dir_list):
    documents = []
    for folder in dir_list:
        folder_path = Path(base_dir) / folder
        for file_path in folder_path.rglob("*.pdf"):
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                documents.extend(docs)
            except Exception as e:
                print(f"[!] Failed to load {file_path}: {e}")
    return documents
