import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


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