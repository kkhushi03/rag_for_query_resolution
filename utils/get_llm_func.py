import yaml
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# load parameters from "params.yaml"
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

EMBEDDING_MODEL = params["EMBEDDING_MODEL"]

def embedding_function():
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )
    return embeddings