import os
from dotenv import load_dotenv
from utils.logger import setup_logger
from utils.config import CONFIG
from data_collection_pipeline.stage_01_create_sources import main as stage_01_main
from data_collection_pipeline.stage_02_open_access import main as stage_02_main
from data_collection_pipeline.stage_03_api_access import fetch_arxiv_papers, search_keywords, api_sources, COLLECTED_DATA_DIR, ARXIV_DIR, download_from_api, scrape_and_download
from data_collection_pipeline.stage_04_download_from_scraped_api_access import main as stage_04_main
from utils.data_collection.file_categorizer import categorize_files


# setup logging
LOG_PATH = CONFIG["LOG_PATH"]
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "pipeline_01_data_collection.log")

def run_data_collection_pipeline():
    logger = setup_logger("data_collection_logger", LOG_FILE)
    
    logger.info("[Part 01] Creating JSON data source....")
    try:
        stage_01_main()
        logger.info("[Part 01] completed successfully.")
    except Exception as e:
        logger.exception(f"[Part 01] failed: {e}")
        return

    logger.info("[Part 02] Scraping OPEN-access sources....")
    try:
        stage_02_main()
        logger.info("[Part 02] completed successfully.")
    except Exception as e:
        logger.exception(f"[Part 02] failed: {e}")
        return

    logger.info("[Part 03] Fetching API-access data....")
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
                logger.info(f"[✓] {name} API accessed successful.")
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
        logger.info("[Part 03] completed successfully.")
    except Exception as e:
        logger.exception(f"[Part 03] failed: {e}")
        return

    logger.info("[Part 04] Downloading from scraped JSON....")
    try:
        stage_04_main()
        logger.info("[✓] Part 04] completed successfully.")
    except Exception as e:
        logger.exception(f"[Part 04] failed: {e}")
        return

if __name__ == "__main__":
    run_data_collection_pipeline()