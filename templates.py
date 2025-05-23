import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "rag_query_resolution"

list_of_files = [
    "data/.gitkeep",
    "chromadb/.gitkeep",
    "artifacts/results/.gitkeep",
    "notebooks/trials.ipynb",
    
    "training_pipeline/__init__.py",
    "training_pipeline/stage_01_populate_db.py",
    "training_pipeline/stage_02_query_data.py",
    
    "utils/__init__.py",
    "utils/logger.py", # for logging
    "utils/get_llm_func.py", # for generation & embedding models
    "utils/get_retrieval_eval_metrics.py", # for retrieval evaluation metrics
    "utils/get_generation_eval_metrics.py", # for generation evaluation metrics
    
    "main.py",
    "test_rag.py",
    "evaluate_rag.py",
    
    "params.yaml",
    "DVC.yaml",
    ".env.local",
    "requirements.txt",
]


for filepath in list_of_files:
    filepath = Path(filepath) #to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")