# rag_for_query_resolution
* rag for enterprise support systems, with limited resources

## run?:
1. environment setup →
    * *to create the virtual environment*: ```python -m venv env```
    
    * *to activate the virtual environment*: ```.\env\Scripts\activate```
    
    * *to deactivate the virtual environment*: ```.\env\Scripts\deactivate```

2. install dependencies → ```pip install -r requirements.txt```

*note*: now just hope everything works (worship to the god you pray)

1. run training pipeline → ```py main.py```
    * *stage_01_populate_db.py*: 
        1. loads documents, 
        2. splits them into several chunks, 
        3. creates a vector db, 
        4. calculates the number of chunks, 
        5. & then finally, those chunks are saved to the vector db (processed in batches for better memory management).
    * *stage_02_query_data.py*:
        1. load the existing db,
        2. search/query the db (using the `similarity score`),
        3. returns context,
        4. prompts the LLM using the prompt template (which includes prompt, question, & context),
        5. generates response,
        6. formats the generated response (by removing all the unnecessary info.)
        7. & then finally, returns the final formatted response with appropriate sources.

2. test rag → ```pytest test_rag.py```

3. run evaluation pipeline → ```py evaluate_rag.py```


## Model info (as of 7th June, 2025):

1. Encoding:
    1. "cl100k_base"

2. Embedding form [MTEB](https://huggingface.co/spaces/mteb/leaderboard):
| Model Name                                | MTEB Rank | M/M Usage (MB) | Params. | Embedding Dim. | Max. Tokens | STS   | Retrieval | Re-ranking |
|-------------------------------------------|-----------|----------------|---------|----------------|-------------|-------|-----------|------------|
| "sentence-transformers/all-MiniLM-L6-v2"  | 118       | 87             | 22M     | 384            | 256         | 56.08 | 32.51     | 40.28      |
| "BAAI/bge-small-en"                       | 87        | 127            | 33M     | 512            | 512         | 59.73 | 36.26     | 45.89      |
| "intfloat/e5-small-v2"                    | 81        | 127            | 33M     | 384            | 512         | 59.87 | 39.38     | 44.44      |
| "intfloat/multilingual-e5-large-instruct" | 7         | 1068           | 560M    | 1024           | 514         | 76.81 | 57.12     | 62.61      |
| "Qwen/Qwen3-Embedding-0.6B"               | 4         | 2272           | 595M    | 1024           | 32768       | 76.71 | 64.65     | 61.41      |

3. Re-ranking:
   1. "cross-encoder/ms-marco-MiniLM-L-6-v2"

4. Generation:
   1. "llama3.2"
        * 3B params