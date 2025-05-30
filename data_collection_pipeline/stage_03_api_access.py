import os
import time
from dotenv import load_dotenv
from utils.download_utils import download_from_api, download_arxiv_papers, search_keywords
from utils.file_categorizer import categorize_files

load_dotenv()

api_sources = {
    "FAO": {
        "endpoint": "https://fenixservices.fao.org/faostat/api/v1/en/resources",
        "api_key_env": "FAO_API_KEY"
    },
    "IEA": {
        "endpoint": "https://api.iea.org/stats/indicator",
        "api_key_env": "IEA_API_KEY"
    },
    "WorldBank": {
        "endpoint": "http://api.worldbank.org/v2/en/indicator",
        "api_key_env": "WORLD_BANK_API_KEY"
    }
    # Add more as needed
}

for name, info in api_sources.items():
    print(f"[+] Downloading via API: {name}")
    api_key = os.getenv(info["api_key_env"])
    download_from_api(info["endpoint"], api_key, base_dir="collected_data")

# def fetch_arxiv_papers(keywords, max_results=50, base_dir="collected_data/arxiv"):
#     print("[+] Starting arXiv paper collection...\n")
#     os.makedirs(base_dir, exist_ok=True)

#     for i, keyword in enumerate(keywords, 1):
#         try:
#             print(f"[{i}/{len(keywords)}] Fetching papers for keyword: '{keyword}'")
#             download_arxiv_papers(keyword, max_results=max_results, base_dir=base_dir)

#             # Sleep to prevent hitting rate limits
#             time.sleep(3)
#         except Exception as e:
#             print(f"  [!] Error while processing '{keyword}': {e}")

#     print("\n[✓] arXiv paper fetch complete.")

# fetch_arxiv_papers(search_keywords, max_results=50)

# Categorize downloaded files
categorize_files("collected_data")