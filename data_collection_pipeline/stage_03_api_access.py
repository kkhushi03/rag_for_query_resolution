import os
from dotenv import load_dotenv
from utils.download_utils import download_from_api
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
    download_from_api(info["endpoint"], api_key, base_dir="outputs")

categorize_files("outputs")