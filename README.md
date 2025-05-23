# rag_for_query_resolution
training & evaluating rag pipeline, when resources are limited

## run?:
1. environment setup →
    * *to create the virtual environment*: ```python -m venv env```
    
    * *to activate the virtual environment*: ```.\env\Scripts\activate```
    
    * *to deactivate the virtual environment*: ```.\env\Scripts\deactivate```

2. install dependencies → ```pip install -r requirements.txt```

*note*: now just hope everything works (worship to the god you pray)

1. run training pipeline → ```py main.py```
    * *it'll run the pipeline one-by-one*
    * *stage_01_populate_db.py*: 
        1. loads documents, 
        2. splits them into several chunks, 
        3. creates a vector db, 
        4. calculates the number of chunks, 
        5. & then finally those chunks are saved to the vector db (using SQL lite in dev stage) in batches
    * *stage_02_query_data.py*:
        1. load the existing db,
        2. search/query the db (using `similarity score`),
        3. returns context,
        4. prompts the LLM the using prompt template (which includes prompt, question, & context),
        5. generates response,
        6. formats the generated response (be removing all the unnecessary info.)
        7. & then finally, returns the formatted final response with appropriate sources

2. test rag → ```pytest test_rag.py```

3. run evaluation pipeline → ```py evaluate_rag.py```