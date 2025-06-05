from utils.config import CONFIG
import tiktoken
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


ENCODING_MODEL = CONFIG["ENCODING_MODEL"]
EMBEDDING_MODEL = CONFIG["EMBEDDING_MODEL"]
LOCAL_LLM = CONFIG["LOCAL_LLM"]


def encoding_func():
    encoding = tiktoken.get_encoding("cl100k_base")
    return encoding

def num_tokens(text: str) -> int:
    encoding = encoding_func()
    # returns the no. of tokens in a text
    return len(encoding.encode(text))

def embedding_func():
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )
    return embeddings

def llm_func(prompt):
    llm = OllamaLLM(
        model=LOCAL_LLM,
        format="json",
        temperature=0.1,
        max_tokens=512,
        # streaming=True,
        streaming=False,
        verbose=True,
    )
    return llm.invoke(prompt)