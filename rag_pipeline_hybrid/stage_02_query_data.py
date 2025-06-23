import os, traceback, pickle
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger
from langchain_community.vectorstores import FAISS
from utils.get_llm_func import embedding_func, rerank_results, llm_func, llm_gen_func
from utils.hybrid_search import HybridRetriever
from utils.get_prompt_temp import prompt_retrieval_grader, prompt_generate_answer
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


# configurations
LOG_PATH = Path(CONFIG["LOG_PATH"])
HYBRID_FAISS_DB_DIR = CONFIG["HYBRID_FAISS_DB_DIR"]
HYBRID_BM25_DB_PATH = Path(CONFIG["HYBRID_BM25_DB_PATH"])

BM25_WEIGHT = CONFIG["BM25_WEIGHT"]
FAISS_WEIGHT = CONFIG["FAISS_WEIGHT"]
HYBRID_TOP_K = CONFIG["HYBRID_TOP_K"]

BATCH_SIZE = CONFIG["BATCH_SIZE"]
K = CONFIG["K"]
R = CONFIG["R"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "hybrid_stage_02_query_rag.log")


def query_rag(query_text: str, faiss_db_dir, bm25_db_path, faiss_weight, bm25_weight, at_k, at_r, logger):
    try:
        try:
            logger.info("[Stage 02, Part 01] Querying FAISS DB.....")
            
            # Load the existing FAISS db (prep the db)
            faiss_db = FAISS.load_local(
                faiss_db_dir,
                embedding_func(),
                allow_dangerous_deserialization=True
            )
            logger.info(f"[Stage 02, Part 01.1] Loading existing FAISS DB from path: {faiss_db_dir}.....")
        except Exception as e:
            logger.error(f"[Stage 02, Part 01] Error loading FAISS DB: {e}")
            logger.debug(traceback.format_exc())
            return []
        
        try:
            logger.info("[Stage 02, Part 02] Loading BM25 database and initializing hybrid retriever...")
            
            hybrid_retriever = HybridRetriever(bm25_weight, faiss_weight)
            
            # Load BM25 index
            if not bm25_db_path.exists():
                logger.error(f"[Stage 02, Part 02] BM25 database not found at: {bm25_db_path}")
                return []
            
            with open(bm25_db_path, "rb") as f:
                bm25_data = pickle.load(f)
            
            # Restore BM25 index components
            hybrid_retriever.bm25_index = bm25_data['bm25_index']
            hybrid_retriever.tokenized_corpus = bm25_data['tokenized_corpus']
            hybrid_retriever.documents = bm25_data['documents']
            
            logger.info(f"[Stage 02, Part 02.1] Successfully loaded BM25 index with {len(hybrid_retriever.documents)} documents")
            
        except Exception as e:
            logger.error(f"[Stage 02, Part 02] Error loading BM25 database: {e}")
            logger.debug(traceback.format_exc())
            return []
        
        try:
            logger.info(f"[Stage 02, Part 03] Performing hybrid search with query: {query_text}")
            
            # Perform hybrid search
            hybrid_results = hybrid_retriever.hybrid_search(
                faiss_db=faiss_db,
                query=query_text,
                top_k=at_k,
            )
            
            logger.info(f"[Stage 02, Part 03.1] Retrieved {len(hybrid_results)} documents from hybrid search:")
            for i, (doc, score) in enumerate(hybrid_results):
                logger.info(f"  [{i}] Hybrid Score: {score:.4f}, Chunk ID: {doc.metadata.get('chunk_id')}")
            
            logger.debug(f"[Result A] Top {at_k} hybrid search results: {hybrid_results}")
            logger.info("[Stage 02, Part 03] Hybrid search completed successfully")
            
        except Exception as e:
            logger.error(f"[Stage 02, Part 03] Error in hybrid search: {e}")
            logger.debug(traceback.format_exc())
            return []
        
        try:
            logger.info(f"[Stage 02, Part 02] Re-ranking {len(hybrid_results)} retrieved results.....")
            
            reranked_results = rerank_results(query_text, hybrid_results)
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
            'results': hybrid_results,
            'rereranked_results': reranked_results,
            'graded_results': graded_results
        }

    except Exception as e:
        logger.error(f"Error in querying FAISS DB & grading: {e}")
        logger.debug(traceback.format_exc())
        return {
            'llms_response': '',
            'context': '',
            'sources': [],
            'results': [],
            'rereranked_results': [],
            'graded_results': []
        }

def run_query_rag(query: str, faiss_db_dir=HYBRID_FAISS_DB_DIR, bm25_db_path=HYBRID_BM25_DB_PATH, faiss_weight=FAISS_WEIGHT, bm25_weight=BM25_WEIGHT, at_k=HYBRID_TOP_K, at_r=R) -> str:
    try:
        logger = setup_logger("query_hybrid_rag_logger", LOG_FILE)
        logger.info(" ")
        logger.info("++++++++Starting Hybrid Querying, Retrieval Grading, & Generation stage....")
        
        if query is None or not query.strip():
            logger.error("[Stage 02] No valid query provided. Exiting.")
            return {
                'llms_response': 'No query provided.',
                'context': '',
                'sources': [],
                'results': [],
                'reranked_results': [],
                'graded_results': []
            }
        
        results = query_rag(query, faiss_db_dir, bm25_db_path, faiss_weight, bm25_weight, at_k, at_r, logger)
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
        logger.debug(f"[Stage 02] Hybrid Retrieved Docs: {results['results']}")
        logger.debug(f"[Stage 02] Reranked Retrieved Docs: {results['rereranked_results']}")
        logger.debug(f"[Stage 02] Graded Re-ranked Retrieved Docs: {results['graded_results']}")
        logger.info(" ")
        
        logger.info("++++++++Hybrid Querying, Retrieval Grading, & Generation stage completed successfully.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error at [Stage 02]: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    run_query_rag(query=None)