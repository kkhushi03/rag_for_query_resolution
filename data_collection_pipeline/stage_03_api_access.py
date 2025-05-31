import os
import time
from dotenv import load_dotenv
from utils.data_collection.download_utils import download_from_api, scrape_and_download, download_arxiv_papers, search_keywords
from utils.data_collection.file_categorizer import categorize_files
from utils.config import CONFIG

load_dotenv()

api_sources = {
    "CORE": {
        "endpoint": "https://api.core.ac.uk/v3/search/works",
        "api_key_env": "CORE_API_KEY"  # Required
    },
}

COLLECTED_DATA_DIR = CONFIG["data_collection_paths"]["collected_data_dir"]
ARXIV_DIR = CONFIG["data_collection_paths"]["arxiv_dir"]

for name, info in api_sources.items():
    print(f"[+] Downloading via API: {name}")
    api_key = os.getenv(info.get("api_key_env", "")) if info.get("api_key_env") else None
    
    try:
        download_from_api(
            endpoint=info["endpoint"],
            api_key=api_key,
            base_dir=COLLECTED_DATA_DIR,
            name=name,
        )
    except Exception as api_error:
        print(f"[✗] {name} API failed: {api_error}")

        fallback_url = info.get("fallback_url")
        if fallback_url:
            print(f"[→] Fallback scraping: {fallback_url}")
            scrape_and_download(
                url=fallback_url,
                base_dir=os.path.join(COLLECTED_DATA_DIR, name.lower()),
                source_name=name
            )
        else:
            print(f"[!] No fallback scraping URL configured for {name}")

def fetch_arxiv_papers(keywords, max_results=50, base_dir=ARXIV_DIR):
    print("[+] Starting arXiv paper collection...\n")
    os.makedirs(base_dir, exist_ok=True)

    for i, keyword in enumerate(keywords, 1):
        try:
            print(f"[{i}/{len(keywords)}] Fetching papers for keyword: '{keyword}'")
            download_arxiv_papers(keyword, max_results=max_results, base_dir=base_dir)

            # Sleep to prevent hitting rate limits
            time.sleep(3)
        except Exception as e:
            print(f"  [!] Error while processing '{keyword}': {e}")

    print("\n[✓] arXiv paper fetch complete.")

fetch_arxiv_papers(search_keywords, max_results=50)

# Categorize downloaded files
categorize_files(COLLECTED_DATA_DIR)