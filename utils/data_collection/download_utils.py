import os
import requests
import json
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote
from utils.data_collection.helpers import use_selenium_to_get_links, infer_year_from_filename, infer_topic_from_filename
import warnings

warnings.filterwarnings("ignore")  # for verify=False SSL bypass

HEADERS = {"User-Agent": "Mozilla/5.0"}


def scrape_and_download(url, base_dir, source_name="Unknown Source"):
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

def download_from_api(endpoint, api_key, base_dir, headers=None, params=None, name="API", max_retries=5):
    os.makedirs(base_dir, exist_ok=True)

    headers = headers or {}
    params = params or {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    attempt = 0
    backoff = 1  # Initial delay in seconds
    while attempt < max_retries:
        try:
            response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            file_name = f"{name.lower()}_data_{int(time.time())}.json"
            path = os.path.join(base_dir, file_name)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=2)

            print(f"[✓] {name}: Data saved to {file_name}")
            return  # Exit after success
        
        except requests.exceptions.RequestException as e:
            print(f"[!] {name}: Attempt {attempt + 1} failed - {e}")

            if attempt < max_retries - 1:
                sleep_time = backoff + random.uniform(0, 0.5)  # Jitter to reduce collision
                print(f"    ↪ Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                backoff *= 2  # Exponential increase
            else:
                print(f"[✗] {name}: Failed after {max_retries} attempts.")
                raise Exception(f"{name} API failed after {max_retries} retries.")

        attempt += 1

# Define your keywords for arXiv
search_keywords = [
    "AI agriculture",
    "climate smart farming",
    "renewable energy forecasting",
    "environmental monitoring with ML",
    "sustainable agriculture",
    "agroecology",
    "precision agriculture",
    "greenhouse gas emissions",
    "smart irrigation systems",
    "crop yield prediction",
    "soil health monitoring",
    "agricultural robotics",
    "drones in agriculture",
]

def download_arxiv_papers(keyword, max_results=30, base_dir=None):
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=all:{quote(keyword)}&start=0&max_results={max_results}"
    url = base_url + query

    print(f"[+] Searching arXiv for: {keyword}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"  [!] Failed to fetch arXiv query: {e}")
        return

    soup = BeautifulSoup(response.content, "xml")
    entries = soup.find_all("entry")
    metadata_list = []

    os.makedirs(base_dir, exist_ok=True)

    for entry in entries:
        title = entry.title.text.strip().replace("\n", " ")
        pdf_link = ""
        for link in entry.find_all("link"):
            if link.get("type") == "application/pdf":
                pdf_link = link.get("href")
                break

        if not pdf_link:
            continue

        file_name = pdf_link.split("/")[-1] + ".pdf"
        file_path = os.path.join(base_dir, file_name)

        if os.path.exists(file_path):
            print(f"  [~] Skipped (already exists): {file_name}")
            metadata_list.append({
                "filename": file_name,
                "title": title,
                "keyword": keyword,
                "url": pdf_link,
                "source": "arXiv",
                "scrape_status": "✅ (cached)"
            })
            continue

        try:
            start_time = time.perf_counter()
            pdf_response = requests.get(pdf_link, headers=HEADERS, timeout=20)
            duration = time.perf_counter() - start_time
            pdf_response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(pdf_response.content)

            print(f"  [+] Downloaded: {file_name} ({duration:.2f}s)")
            metadata_list.append({
                "filename": file_name,
                "title": title,
                "keyword": keyword,
                "url": pdf_link,
                "source": "arXiv",
                "scrape_status": "✅"
            })

            # Sleep to avoid rate limits
            time.sleep(1.5)

        except Exception as e:
            print(f"  [!] Failed to download: {file_name} → {e}")
            metadata_list.append({
                "filename": file_name,
                "title": title,
                "keyword": keyword,
                "url": pdf_link,
                "source": "arXiv",
                "scrape_status": "❌"
            })

    # Save metadata
    if metadata_list:
        meta_path = os.path.join(base_dir, "arxiv_meta.jsonl")
        with open(meta_path, "a", encoding="utf-8") as f:
            for entry in metadata_list:
                f.write(json.dumps(entry) + "\n")