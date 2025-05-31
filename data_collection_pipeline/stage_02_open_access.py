import os
import json
from utils.data_collection.download_utils import scrape_and_download
from utils.data_collection.file_categorizer import categorize_files
from utils.config import CONFIG


def load_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Loaded JSON from {json_path}")
        return data
    except FileNotFoundError:
        print(f"[ERROR] File not found: {json_path}")
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to decode JSON: {json_path}")
    except Exception as e:
        print(f"[ERROR] Unexpected error while reading JSON: {e}")
    return {}

def scrape_sources(data, output_dir):
    if not data.get("scrapable_or_direct_download"):
        print("[WARNING] No scrapable sources found in data.")
        return
    
    for source in data["scrapable_or_direct_download"]:
        source_name = source.get("name", "Unknown Source")
        print(f"[INFO] Scraping: {source_name}")
        sub_links = source.get("sub_links", [])
        if not sub_links:
            print(f"[WARNING] No sub_links found for {source_name}")
            continue
        for link in sub_links:
            try:
                scrape_and_download(link, base_dir=output_dir, source_name=source_name)
                print(f"[SUCCESS] Scraped: {link}")
            except Exception as e:
                print(f"[ERROR] Failed to scrape {link}: {e}")

def categorize_downloaded_files(output_dir):
    try:
        categorize_files(output_dir)
        print(f"[INFO] Files in '{output_dir}' categorized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to categorize files: {e}")

def main():
    json_path = CONFIG["data_collection_paths"]["data_sources_json"]
    output_dir = CONFIG["data_collection_paths"]["collected_data_dir"]

    data = load_json(json_path)
    if data:
        scrape_sources(data, output_dir=output_dir)
        categorize_downloaded_files(output_dir=output_dir)

if __name__ == "__main__":
    main()