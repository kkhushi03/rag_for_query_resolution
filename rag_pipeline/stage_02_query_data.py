import os
import traceback
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from langchain_chroma import Chroma
from utils.get_llm_func import embedding_func, llm_func
# from utils.get_prompt_temp import prompt_retrieval_grader, prompt_generate_answer
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# configurations
LOG_PATH = CONFIG["LOG_PATH"]
LOG_PATH = Path(LOG_PATH)
CHROMA_DB_PATH = CONFIG["CHROMA_DB_PATH"]
EVALUATION_DATA_PATHS = CONFIG["EVALUATION_DATA_PATHS"]
EVALUATION_DATA_PATHS = Path(EVALUATION_DATA_PATHS)

BATCH_SIZE = CONFIG["BATCH_SIZE"]
K = CONFIG["K"]
R = CONFIG["R"]
N = CONFIG["N"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_02_query_rag.log")


def query_rag(query_text: str, chroma_db_dir, at_k, logger):
    try:
        logger.info("[Part 01] Querying Chroma DB.....")
        
        # load the existing db (prep the db)
        db = Chroma(
            embedding_function=embedding_func(),
            persist_directory=chroma_db_dir,
        )
        logger.info(f"[Part 01.1] Loading existing DB from path: {chroma_db_dir}")
        
        # query the db (search the db)
        logger.info(f"[Part 01.2] Searching the db with text using similarity search: {query_text}")
        results = db.similarity_search_with_score(query_text, k=at_k)
        
        logger.info(f"[Result A] Found {len(results)} results: {results}")
        logger.info("[Part 01] Querying Chroma DB completed successfully")
        
        return results
    except Exception as e:
        logger.error(f"Error in querying Chroma DB: {e}")
        logger.debug(traceback.format_exc())
        return []
        
    #     logger.info("[Part 02] Grading Retrieved Docs.....")
    #     retrieval_grader = prompt_retrieval_grader | llm_func | JsonOutputParser()
    #     graded_results = []

    #     for i, (doc, score) in enumerate(results):
    #         doc_text = doc.page_content
    #         logger.debug(f"[Part 02.{i}] Grading doc with score {score}")
    #         try:
    #             grade = retrieval_grader.invoke({
    #                 "question": query_text,
    #                 "document": doc_text,
    #             })
    #             graded_results.append((doc, score, grade["score"]))
    #             logger.info(f"[Part 02.{i}] Graded as: {grade['score']}")
    #         except Exception as grading_err:
    #             logger.error(f"[Part 02.{i}] Grading failed: {grading_err}")
    #             logger.debug(traceback.format_exc())

    #     logger.info(f"[Result B] Found {len(graded_results)} graded results: {graded_results}")
    #     logger.info("[Part 02] Grading completed successfully")
        
    #     logger.info("[Part 03] Generating Answer from LLM.....")
    #     context_text = "\n\n---\n\n".join([doc.page_content for doc, _, _ in graded_results])
        
    #     output_parser = StrOutputParser()
    #     response_generator = prompt_generate_answer | llm_func | output_parser
        
    #     response_text = response_generator.invoke({
    #         "question": query_text,
    #         "context": context_text
    #     })
    #     sources = [doc.metadata.get("chunk_id", None) for doc, _, _ in graded_results]
        
    #     logger.info(f"[Result C] LLM's response: {response_text}")
    #     logger.info(f"[Result D] Sources: {sources}")
        
    #     logger.info("[Part 03] Generating Answer from LLM completed successfully")
        
    #     return {
    #         'formatted_question': response_text,
    #         'context': context_text,
    #         'sources': sources,
    #         'graded_results': graded_results
    #     }

    # except Exception as e:
    #     logger.error(f"Error in querying Chroma DB & grading: {e}")
    #     logger.debug(traceback.format_exc())
    #     return {
    #         'formatted_question': '',
    #         'context': '',
    #         'sources': [],
    #         'graded_results': []
    #     }

def run_query_rag(query: str, chroma_db_dir=CHROMA_DB_PATH, at_k=K) -> str:
    try:
        logger = setup_logger("query_rag_logger", LOG_FILE)
        logger.info(" ")
        logger.info("++++++++Starting Querying, Retrieval Grading, & Generation stage....")
        
        results = query_rag(query, chroma_db_dir, at_k, logger)
        if query is None:
            logger.error("No query provided. Exiting.")
            return
        logger.debug(f"Results: {results}")

        # logger.info("Graded results:")
        # for idx, (doc, sim_score, relevance) in enumerate(results["graded_results"]):
        #     logger.info(f"[Doc {idx}] Score: {sim_score:.4f}, Relevant: {relevance}")
        #     logger.debug(f"Content: {doc.page_content[:300]}...")
        
        # logger.info(f"LLM's response: {results['formatted_question']}")
        # logger.info(f"Sources: {results['sources']}")
        # logger.info(f"Context: {results['context'][:300]}...")
        
        logger.info("++++++++Querying, Retrieval Grading, & Generation stage completed successfully.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_query_rag(query=None)