import os, traceback, re, json, pickle
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from utils.get_llm_func import rerank_results, llm_func, llm_gen_func
from sklearn.metrics.pairwise import cosine_similarity
from utils.get_prompt_temp import prompt_retrieval_grader, prompt_generate_answer
from langchain.schema import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
CHUNKS_OUT_PATH_TFIDF = Path(CONFIG["CHUNKS_OUT_PATH_TFIDF"])
TFIDF_DB_DIR = Path(CONFIG["TFIDF_DB_DIR"])
TFIDF_DB_PATH = Path(CONFIG["TFIDF_DB_PATH"])
TFIDF_META_PATH = Path(CONFIG["TFIDF_META_PATH"])

BATCH_SIZE = CONFIG["BATCH_SIZE"]
K = CONFIG["K"]
R = CONFIG["R"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_02_query_rag.log")


def query_rag(query_text: str, tfidf_db_dir, tfidf_db_path, tfidf_metadata_path, at_k, at_r, logger):
    try:
        try:
            logger.info("[Stage 02, Part 01] Querying TF-IDF DB.....")
            
            # Load the existing TF-IDF db (prep the db)
            if tfidf_db_dir.exists():
                with open(tfidf_db_path, "rb") as f:
                    tfidf, vectors, texts = pickle.load(f)

                with open(tfidf_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                logger.info(f"[Stage 02, Part 01.1] Loading existing TF-IDF DB from path: {tfidf_db_dir}.....")
            else:
                logger.error(f"[Stage 02, Part 01.1] TF-IDF DB not found at path: {tfidf_db_dir}. Please populate the DB first.")
                return []
            
            # reconstruct document objects
            documents = [
                Document(page_content=texts[i], metadata=metadata[i])
                for i in range(len(texts))
            ]

            # query the db (search the db)
            logger.info(f"[Stage 02, Part 01.2] Searching the db with text using similarity search: {query_text}.....")
            def tfidf_similarity_search_with_score(query_text, tfidf, vectors, documents, at_k):
                query_vector = tfidf.transform([query_text])
                sims = cosine_similarity(query_vector, vectors).flatten()
                top_k_idx = sims.argsort()[-at_k:][::-1]
                return [(documents[i], sims[i]) for i in top_k_idx]

            results = tfidf_similarity_search_with_score(query_text, tfidf, vectors, documents, at_k)
            logger.info(f"[Stage 02, Part 01.3] Retrieved {len(results)} Docs:")

            for i, (doc, sim_score) in enumerate(results):
                logger.info(f"  [{i}] Similarity Score: {sim_score:.4f}, Chunk ID: {doc.metadata.get('chunk_id')}")
            logger.debug(f"[Result A] Top {at_k} retrieved results: {results}")
            
            logger.info("[Stage 02, Part 01] Querying TF-IDF DB completed successfully")
        except Exception as e:
            logger.error(f"[Stage 02, Part 01] Error in querying TF-IDF DB: {e}")
            logger.debug(traceback.format_exc())
            return []
        
        try:
            logger.info(f"[Stage 02, Part 02] Re-ranking {len(results)} retrieved results.....")
            
            reranked_results = rerank_results(query_text, results)
            for i, (doc, score) in enumerate(reranked_results):
                logger.info(f"  [{i}] Re-rank Score: {score:.4f}, Chunk ID: {doc.metadata.get('chunk_id')}")
            
            reranked_results = reranked_results[:at_r]
            logger.debug(f"[Result B] Top {at_r} after re-ranking: {reranked_results}")
            
            logger.info("[Stage 02, Part 02] Re-ranking completed successfully")
        except Exception as e:
            logger.error(f"[Stage 02, Part 02] Error in re-ranking retrieved results: {e}")
            logger.debug(traceback.format_exc())
            return []
        
        try:
            logger.info(f"[Stage 02, Part 03] Grading Top {len(reranked_results)} Re-ranked Retrieved Docs.....")
            retrieval_grader = prompt_retrieval_grader | llm_func | JsonOutputParser()
            graded_results = []

            for i, (doc, score) in enumerate(reranked_results):
                doc_text = doc.page_content
                logger.debug(f"  [{i}] Grading doc with score: {score}")
                try:
                    grade = retrieval_grader.invoke({
                        "question": query_text,
                        "document": doc_text,
                    })
                    graded_results.append((doc, score, grade["score"]))
                    logger.info(f"  [{i}] Graded: {grade['score']}, Chunk ID: {doc.metadata.get('chunk_id')}")
                except Exception as grading_err:
                    logger.error(f"  [{i}] Grading failed: {grading_err}")
                    logger.debug(traceback.format_exc())
                    continue
            
            logger.debug(f"[Result C] Top {at_r} results after grading: {graded_results}")
            logger.info(f"[Stage 02, Part 03.1] Filtering only 'YES' results.....")
            graded_results = [item for item in graded_results if item[2] == "YES"]
            logger.info(f"[Stage 02, Part 03.2] Number of results after filtering only 'YES': {len(graded_results)}")
            
            logger.info("[Stage 02, Part 03] Grading completed successfully.")
        except Exception as e:
            logger.error(f"[Stage 02, Part 03] Error in grading retrieved results: {e}")
            logger.debug(traceback.format_exc())
            return []
        
        try:
            logger.info(f"[Stage 02, Part 04] Generating Answer with LLM from {len(graded_results)} graded documents.....")
            if not graded_results:
                logger.warning("No graded documents with 'YES' passed. Skipping generation.")
                return []
            
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _, _ in graded_results])
            if not context_text.strip():
                logger.warning("Empty context_text! Cannot generate answer.")
                return []
            
            response_generator = prompt_generate_answer | llm_gen_func | StrOutputParser()
            
            try:
                response_text = response_generator.invoke({
                    "question": query_text,
                    "context": context_text
                })
                # response_text = response_text.get("answer", "")
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response_text = ""
            
            logger.debug(f"[Result D] LLM's response: {response_text}")
            
            sources = [doc.metadata.get("chunk_id", None) for doc, _, _ in graded_results]
            logger.debug(f"[Result E] Sources: {sources}")
            
            logger.info("[Stage 02, Part 04] Generating Answer from LLM completed successfully")
        except Exception as e:
            logger.error(f"[Stage 02, Part 04] Error in generating answer from LLM: {e}")
            logger.debug(traceback.format_exc())
            return []
        
        return {
            'llms_response': response_text,
            'context': context_text,
            'sources': sources,
            'results': results,
            'rereranked_results': reranked_results,
            'graded_results': graded_results
        }

    except Exception as e:
        logger.error(f"Error in querying TFIDF DB & grading: {e}")
        logger.debug(traceback.format_exc())
        return {
            'llms_response': '',
            'context': '',
            'sources': [],
            'results': [],
            'rereranked_results': [],
            'graded_results': []
        }

def run_query_rag(query: str, tfidf_db_dir=TFIDF_DB_DIR, tfidf_db_path=TFIDF_DB_PATH, tfidf_metadata_path=TFIDF_META_PATH, at_k=K, at_r=R) -> str:
    try:
        logger = setup_logger("query_rag_logger", LOG_FILE)
        logger.info(" ")
        logger.info("++++++++Starting Querying, Retrieval Grading, & Generation stage....")
        
        results = query_rag(query, tfidf_db_dir, tfidf_db_path, tfidf_metadata_path, at_k, at_r, logger)
        if query is None:
            logger.error("[Stage 02] No query provided. Exiting.")
            return
        logger.debug(f"[Stage 02] Results: {results}")
        
        logger.info(" ")
        logger.info("#--#--FINAL RESULTS:--#--#")
        logger.info(" ")
        logger.info(f"[Stage 02] Query: {query}")
        logger.info(" ")
        logger.info(f"[Stage 02] LLM's response: {results['llms_response']}")
        logger.info(" ")
        logger.info(f"[Stage 02] Sources: {results['sources']}")
        logger.info(" ")
        logger.info(f"[Stage 02] Context: {results['context']}")
        logger.info(" ")
        logger.debug(f"[Stage 02] Retrieved Docs: {results['results']}")
        logger.debug(f"[Stage 02] Reranked Retrieved Docs: {results['rereranked_results']}")
        logger.debug(f"[Stage 02] Graded Re-ranked Retrieved Docs: {results['graded_results']}")
        logger.info(" ")
        
        logger.info("++++++++Querying, Retrieval Grading, & Generation stage completed successfully.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error at [Stage 02]: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_query_rag(query=None)