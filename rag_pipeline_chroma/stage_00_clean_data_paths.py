import os
import re
import shutil
import traceback
from pathlib import Path
from utils.config import CONFIG
from utils.logger import setup_logger


# configuration
LOG_PATH = CONFIG["LOG_PATH"]
LOG_PATH = Path(LOG_PATH)
SOURCE_DATA_PATH = CONFIG["SOURCE_DATA_PATH"]
SOURCE_DATA_PATH = Path(SOURCE_DATA_PATH)
SOURCE_DIRS = CONFIG["SOURCE_DIRS"]
DATA_PATH = CONFIG["DATA_PATH"]
DATA_PATH = Path(DATA_PATH)

BATCH_SIZE = CONFIG["BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_00_clean_data_paths.log")


def clean_filename(name: str, logger) -> str:
    logger.info(f"[Part 01] Cleaning filename (by removing unwanted characters and spaces): {name}")
    name = name.strip()
    
    # Split extension (e.g., ".pdf") and clean only the name part
    stem, ext = os.path.splitext(name)
    
    # remove special characters except dashes, underscores, dots
    # name = re.sub(r'[^\w\s\-\.]', '', name)
    cleaned_stem = re.sub(r'[^\w\s\-]', '', stem)  # remove all except word chars, dash, space
    # replace spaces with underscores
    # name = re.sub(r'\s+', '_', name)
    cleaned_stem = re.sub(r'\s+', '_', cleaned_stem)
    
    # add extension back
    name = cleaned_stem + ext
    
    logger.info(f"[Part 02] Cleaned filename: {name}")
    return name

def gather_pdfs(source_dirs, source_data_path, logger):
    all_pdfs = []
    logger.info("[Part 03] Collecting PDFs from source directories...")
    
    try:
        for folder in source_dirs:
            source_path = source_data_path / folder
            pdfs = list(source_path.rglob("*.pdf"))
            logger.debug(f"Found {len(pdfs)} PDFs in {folder}")
            all_pdfs.extend(pdfs)
        logger.info(f"[Part 04] Total PDFs found: {len(all_pdfs)}")
        return all_pdfs
    except Exception as e:
        logger.error(f"Failed to collect PDFs: {e}")
        logger.debug(traceback.format_exc())
        return []

def group_and_copy_pdfs(pdfs, target_data_path, batch_size, logger):
    logger.info("[Part 05] Creating target data path...")
    target_data_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Part 06] Grouping PDFs into folders of {batch_size} each and copying them with cleaned names...")
    try:
        for i in range(0, len(pdfs), batch_size):
            group_num = (i // batch_size) + 1
            group_dir = DATA_PATH / f"group_{group_num}"
            group_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[Part 06.1] Creating {group_num} number of group directories: {group_dir}")

            for idx, pdf in enumerate(pdfs[i:i + batch_size], start=i + 1):
                clean_name = clean_filename(pdf.stem, logger) + ".pdf"
                target_path = group_dir / clean_name

                logger.info(f"[Part 06.2.{idx}] Copying PDF #{idx}: {pdf.name} to {target_path}")
                try:
                    shutil.copy(pdf, target_path)
                    logger.info(f"[Part 06.3] Copied: {pdf} -> {target_path}")
                except Exception as e:
                    logger.error(f"Failed to copy {pdf}: {e}")
                    logger.debug(traceback.format_exc())
                    continue
    except Exception as e:
        logger.error(f"Failed to group and copy PDFs: {e}")
        logger.debug(traceback.format_exc())
        return

def run_clean_data_paths(source_dirs=SOURCE_DIRS, source_data_path=SOURCE_DATA_PATH, target_data_path=DATA_PATH, batch_size=BATCH_SIZE):
    try:
        logger = setup_logger("clean_data_paths_logger", LOG_FILE)
        logger.info(" ")
        logger.info("--------++++++++Starting data path cleaning stage.....")
        
        pdfs = gather_pdfs(source_dirs, source_data_path, logger)
        if not pdfs:
            logger.error("No PDFs found. Exiting.")
            return
        
        group_and_copy_pdfs(pdfs, target_data_path, batch_size, logger)
        if not pdfs:
            logger.error("No PDFs copied. Exiting.")
            return
        
        logger.info("--------++++++++Data paths cleaning stage successfully completed.")
        logger.info(" ")
    except Exception as e:
        logger.error(f"Error at [Stage 00]: {e}")
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    run_clean_data_paths()
