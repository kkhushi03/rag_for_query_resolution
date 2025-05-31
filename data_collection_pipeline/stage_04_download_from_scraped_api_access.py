import os
import json
import requests
import re
from utils.config import CONFIG

DOWNLOAD_DIR = CONFIG["data_collection_paths"]["collected_data_dir"] + "/core"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

JSON_FILES = CONFIG["data_collection_paths"]["core_json_files"]
COLLECTED_DATA_DIR = CONFIG["data_collection_paths"]["collected_data_dir"]

def load_json_file(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[INFO] Loaded JSON from {json_path}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load {json_path}: {e}")
        return []


def scrape_and_download_from_core(url, base_dir=COLLECTED_DATA_DIR, source_name="CORE", filename_hint=None):
    os.makedirs(f"{base_dir}/{source_name}", exist_ok=True)

    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to download: {url} (Status: {response.status_code})")

    # Determine file extension
    ext = ".pdf" if "pdf" in response.headers.get("Content-Type", "") else ".html"
    filename = filename_hint or "downloaded_file"
    filepath = os.path.join(base_dir, source_name, filename + ext)

    with open(filepath, "wb") as f:
        f.write(response.content)

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '', filename)

def extract_and_download(json_data, output_dir=COLLECTED_DATA_DIR, source_name="CORE"):
    if not isinstance(json_data, list):
        print("[WARNING] Expected a list of papers but got something else. Skipping...")
        return

    print(f"[INFO] Found {len(json_data)} entries. Looking for download URLs...")

    for idx, paper in enumerate(json_data):
        url = paper.get("downloadUrl")
        # title = paper.get("title", f"paper_{idx}")
        # title_cleaned = sanitize_filename(title.strip().replace(" ", "_"))[:100]
        title = paper.get("title") or f"paper_{idx}"
        title_cleaned = sanitize_filename(title.strip().replace(" ", "_"))
        if not title_cleaned:
            title_cleaned = f"paper_{idx}"
        title_cleaned = title_cleaned[:100]

        if url and url.endswith(".pdf"):
            try:
                scrape_and_download_from_core(url, base_dir=output_dir, source_name=source_name, filename_hint=title_cleaned)
                print(f"[✓] Downloaded: {title_cleaned}")
            except Exception as e:
                print(f"[✗] Failed to download {url}: {e}")
        else:
            print(f"[INFO] Skipped: No valid PDF download URL for entry {idx}")


def main():
    for json_file in JSON_FILES:
        data = load_json_file(json_file)

        print(f"[DEBUG] Type of loaded data: {type(data)}")
        if isinstance(data, dict):
            print(f"[DEBUG] Top-level keys: {list(data.keys())}")
            # Access list of papers inside 'results'
            if "results" in data and isinstance(data["results"], list):
                extract_and_download(data["results"], output_dir=COLLECTED_DATA_DIR, source_name="CORE")
            else:
                print("[WARNING] No recognized paper list key found in dict. Skipping...")
        elif isinstance(data, list):
            extract_and_download(data, output_dir=COLLECTED_DATA_DIR, source_name="CORE")
        else:
            print("[WARNING] Unsupported JSON structure. Skipping...")

if __name__ == "__main__":
    main()
