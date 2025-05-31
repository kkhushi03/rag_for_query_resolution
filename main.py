import os
from dotenv import load_dotenv
from utils.logger import setup_logger
from data_collection_pipeline.stage_01_create_sources import main as stage_01_main
from data_collection_pipeline.stage_02_open_access import main as stage_02_main
from data_collection_pipeline.stage_03_api_access import fetch_arxiv_papers, search_keywords, api_sources, COLLECTED_DATA_DIR, ARXIV_DIR, download_from_api, scrape_and_download
from data_collection_pipeline.stage_04_download_from_scraped_api_access import main as stage_04_main
from utils.data_collection.file_categorizer import categorize_files
from utils.config import CONFIG


def run_data_collection_pipeline():
    # === Setup logging ===
    LOG_PATH = CONFIG["LOG_PATH"]
    LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "pipeline_01_data_collection.log")

    logger = setup_logger("data_collection_logger", LOG_FILE)

    logger.info(" ")
    logger.info("*******************[Pipeline 1] DATA COLLECTION STARTING*******************")
    logger.info(" ")

    # === Stage 01 ===
    logger.info("*******************[Stage 01] CREATING DATA SOURCE JSON*******************")
    try:
        stage_01_main()
        logger.info("[✓] Stage 01 completed successfully.")
    except Exception as e:
        logger.exception(f"[✗] Stage 01 failed: {e}")
        return

    # === Stage 02 ===
    logger.info("*******************[Stage 02] SCRAPING OPEN ACCESS SOURCES*******************")
    try:
        stage_02_main()
        logger.info("[✓] Stage 02 completed successfully.")
    except Exception as e:
        logger.exception(f"[✗] Stage 02 failed: {e}")
        return

    # === Stage 03 ===
    logger.info("*******************[Stage 03] FETCHING API ACCESS DATA*******************")
    try:

        for name, info in api_sources.items():
            logger.info(f"[→] Accessing API: {name}")
            load_dotenv()
            api_key = os.getenv(info.get("api_key_env", "")) if info.get("api_key_env") else None

            try:
                download_from_api(
                    endpoint=info["endpoint"],
                    api_key=api_key,
                    base_dir=COLLECTED_DATA_DIR,
                    name=name,
                )
                logger.info(f"[✓] {name} API download successful.")
            except Exception as api_error:
                logger.warning(f"[✗] {name} API failed: {api_error}")
                fallback_url = info.get("fallback_url")
                if fallback_url:
                    logger.info(f"[→] Fallback scraping for {name}")
                    try:
                        scrape_and_download(
                            url=fallback_url,
                            base_dir=os.path.join(COLLECTED_DATA_DIR, name.lower()),
                            source_name=name
                        )
                    except Exception as scrape_error:
                        logger.exception(f"[✗] Fallback scraping failed for {name}: {scrape_error}")
                else:
                    logger.warning(f"[!] No fallback scraping URL configured for {name}")

        fetch_arxiv_papers(search_keywords, max_results=50, base_dir=ARXIV_DIR)
        categorize_files(COLLECTED_DATA_DIR)
        logger.info("[✓] Stage 03 completed successfully.")
    except Exception as e:
        logger.exception(f"[✗] Stage 03 failed: {e}")
        return

    # === Stage 04 ===
    logger.info("*******************[Stage 04] DOWNLOADING FROM SCRAPED JSON*******************")
    try:
        stage_04_main()
        logger.info("[✓] Stage 04 completed successfully.")
    except Exception as e:
        logger.exception(f"[✗] Stage 04 failed: {e}")
        return

    logger.info(" ")
    logger.info("*******************[Pipeline 1] DATA COLLECTION COMPLETED*******************")
    logger.info(" ")


if __name__ == "__main__":
    run_data_collection_pipeline()
