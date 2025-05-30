import os
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from utils.helpers import use_selenium_to_get_links, infer_year_from_filename, infer_topic_from_filename
import warnings

warnings.filterwarnings("ignore")  # for verify=False SSL bypass

HEADERS = {"User-Agent": "Mozilla/5.0"}

# def scrape_and_download(url, base_dir="collected_data"):
#     try:
#         res = requests.get(url, timeout=20)
#         res.raise_for_status()
#         soup = BeautifulSoup(res.text, "html.parser")

#         for tag in soup.find_all("a", href=True):
#             href = tag['href']
#             if any(href.endswith(ext) for ext in [".pdf", ".csv", ".json", ".xlsx", ".xls", ".txt", ".pptx", ".md"]):
#                 full_url = urljoin(url, href)
#                 file_name = full_url.split("/")[-1].split("?")[0]
#                 os.makedirs(base_dir, exist_ok=True)
#                 file_path = os.path.join(base_dir, file_name)
#                 with open(file_path, "wb") as f:
#                     f.write(requests.get(full_url).content)
#                 print(f"  - Downloaded: {file_name}")

#     except Exception as e:
#         print(f"  [!] Failed to scrape {url}: {e}")


def scrape_and_download(url, base_dir="collected_data", source_name="Unknown Source"):
    try:
        try:
            res = requests.get(url, headers=HEADERS, timeout=20)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception:
            print(f"  [Fallback] Using Selenium for: {url}")
            soup = use_selenium_to_get_links(url)

        metadata = []
        os.makedirs(base_dir, exist_ok=True)

        for tag in soup.find_all("a", href=True):
            href = tag['href']
            if any(href.endswith(ext) for ext in [".pdf", ".csv", ".json", ".xlsx", ".xls", ".txt", ".pptx", ".md"]):
                full_url = urljoin(url, href)
                file_name = full_url.split("/")[-1].split("?")[0]
                ext = os.path.splitext(file_name)[-1].lower()
                file_path = os.path.join(base_dir, file_name)

                # ✅ Skip download if already exists
                if os.path.exists(file_path):
                    print(f"  [SKIP] Already exists: {file_name}")
                    metadata.append({
                        "filename": file_name,
                        "source": source_name,
                        "file_type": ext.upper().replace('.', ''),
                        "year": infer_year_from_filename(file_name),
                        "topic_keyword": infer_topic_from_filename(file_name),
                        "url": full_url,
                        "scrape_status": "✅ (cached)"
                    })
                    continue

                try:
                    resp = requests.get(full_url, headers=HEADERS, timeout=30, verify=False)
                    resp.raise_for_status()

                    # 🧠 Optional: Warn if file size seems very small (<1KB)
                    if len(resp.content) < 1024:
                        print(f"  [!] Warning: {file_name} may be incomplete (size < 1KB)")

                    with open(file_path, "wb") as f:
                        f.write(resp.content)

                    print(f"  - Downloaded: {file_name}")
                    metadata.append({
                        "filename": file_name,
                        "source": source_name,
                        "file_type": ext.upper().replace('.', ''),
                        "year": infer_year_from_filename(file_name),
                        "topic_keyword": infer_topic_from_filename(file_name),
                        "url": full_url,
                        "scrape_status": "✅"
                    })

                except Exception as download_error:
                    print(f"  [!] Failed to download: {file_name} | Reason: {download_error}")
                    metadata.append({
                        "filename": file_name,
                        "source": source_name,
                        "file_type": ext.upper().replace('.', ''),
                        "year": "Unknown",
                        "topic_keyword": "Unknown",
                        "url": full_url,
                        "scrape_status": "❌"
                    })

        # 💾 Save metadata to a temporary .jsonl file
        meta_path = os.path.join(base_dir, "temp_meta.jsonl")
        with open(meta_path, "a", encoding="utf-8") as f:
            for entry in metadata:
                f.write(json.dumps(entry) + "\n")

    except Exception as e:
        print(f"  [!] Fatal error scraping {url}: {e}")

def download_from_api(endpoint, api_key, base_dir="collected_data"):
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        res = requests.get(endpoint, headers=headers, timeout=20)
        res.raise_for_status()
        file_name = endpoint.split("/")[-1] + ".json"
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, file_name), "w") as f:
            f.write(res.text)
        print(f"  - API data saved: {file_name}")
    except Exception as e:
        print(f"  [!] API request failed: {e}")
