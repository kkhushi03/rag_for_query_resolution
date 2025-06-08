import os
import pandas as pd
import traceback
from typing import List, Dict
from datetime import datetime

def ensure_directory_exists(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def append_new_results_to_csv(results: List[Dict], result_csv_path: str, logger):
    try:
        ensure_directory_exists(result_csv_path)

        # Add `date_time` to each result entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for row in results:
            row['date_time'] = timestamp

        df_new = pd.DataFrame(results)

        if os.path.exists(result_csv_path):
            logger.info(f"Appending to existing CSV at: {result_csv_path}")
            df_existing = pd.read_csv(result_csv_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(result_csv_path, index=False)
            logger.info(f"Appended {len(results)} new rows.")
        else:
            logger.info(f"Creating new CSV at: {result_csv_path}")
            df_new.to_csv(result_csv_path, index=False)
            logger.info(f"Saved {len(results)} rows in new CSV.")

    except Exception as e:
        logger.error(f"Error while saving results to CSV: {e}")
        logger.debug(traceback.format_exc())
