import tiktoken
from utils.config import CONFIG
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaLLM


ENCODING_MODEL = CONFIG["ENCODING_MODEL"]
EMBEDDING_MODEL = CONFIG["EMBEDDING_MODEL"]
RE_RANK_MODEL = CONFIG["RE_RANK_MODEL"]
LOCAL_LLM = CONFIG["LOCAL_LLM"]

cross_encoder = CrossEncoder(RE_RANK_MODEL)


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

def rerank_results(query: str, docs_with_scores):
    pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
    scores = cross_encoder.predict(pairs)
    
    reranked = sorted(
        zip(docs_with_scores, scores),
        key=lambda x: x[1],
        reverse=True  # higher is better
    )
    return [(doc, score) for ((doc, _), score) in reranked]

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

def llm_gen_func(prompt):
    llm = OllamaLLM(
        model=LOCAL_LLM,
        # format="json",
        temperature=0.1,
        max_tokens=512,
        # streaming=True,
        streaming=False,
        verbose=True,
    )
    return llm.invoke(prompt)