import os, argparse, traceback
from utils.logger import setup_logger
from utils.config import CONFIG
# from rag_pipeline_hybrid.stage_01_populate_db import run_populate_db
# from rag_pipeline_hybrid.stage_02_query_data import run_query_rag
from rag_pipeline_hybrid.stage_03_eval_queries import run_evaluation

# configurations
QUERY_TEXT = CONFIG["QUERY_TEXT"]

# setup logging
LOG_PATH = CONFIG["LOG_PATH"]
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "hybrid_main.log")


def main():
    logger = setup_logger("main_logger", LOG_FILE)
    
    # Create CLI.
    parser = argparse.ArgumentParser(description="MAIN WORKFLOW")
    # parser.add_argument("--reset", action="store_true", help="Reset FAISS DB before population")
    # parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    
    try:
        logger.info(" ")
        logger.info("////--//--//----STARTING [PIPELINE 02] RAG PIPELINE----//--//--////")
        
        # try:
        #     logger.info(" ")
        #     logger.info("----------STARTING [STAGE 01] POPULATE DB----------")
        #     run_populate_db(args.reset)
        #     # logger.info("Already Done. Skipping...")
        #     logger.info("----------FINISHED [STAGE 01] POPULATE DB----------")
        #     logger.info(" ")
        # except Exception as e:
        #     logger.error(f"ERROR RUNNING [STAGE 01] POPULATE DB: {e}")
        #     logger.debug(traceback.format_exc())
        #     return
        
        # try:
        #     logger.info(" ")
        #     logger.info("----------STARTING [STAGE 02] QUERY RAG----------")
        #     # run_query_rag(args.query_text)
        #     run_query_rag(query=QUERY_TEXT)
        #     # logger.info("Already Done. Skipping...")
        #     logger.info("----------FINISHED [STAGE 02] QUERY RAG----------")
        #     logger.info(" ")
        # except Exception as e:
        #     logger.error(f"ERROR RUNNING [STAGE 02] QUERY RAG: {e}")
        #     logger.debug(traceback.format_exc())
        #     return
        
        try:
            logger.info(" ")
            logger.info("----------STARTING [STAGE 03] EVALUATE QUERIES----------")
            run_evaluation()
            # logger.info("Already Done. Skipping...")
            logger.info("----------FINISHED [STAGE 03] EVALUATE QUERIES----------")
            logger.info(" ")
        except Exception as e:
            logger.error(f"ERROR RUNNING [STAGE 03] EVALUATE QUERIES: {e}")
            logger.debug(traceback.format_exc())
            return
        
        logger.info("////--//--//----FINISHED [PIPELINE 02] RAG PIPELINE----//--//--////")
        logger.info(" ")
    except Exception as e:
        logger.error(f"ERROR RUNNING [PIPELINE 02] RAG PIPELINE: {e}")
        logger.debug(traceback.format_exc())
        return


if __name__ == "__main__":
    main()
