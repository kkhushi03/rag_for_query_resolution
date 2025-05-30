import os
import shutil
import csv
import json

CATEGORY_MAP = {
    ".pdf": "pdfs",
    ".csv": "csvs",
    ".json": "jsons",
    ".xlsx": "excels",
    ".xls": "excels",
    ".html": "htmls",
    ".htm": "htmls",
    ".md": "markdowns",
    ".txt": "txts",
    ".pptx": "ppts",
    ".ppt": "ppts",
}

# def categorize_files(base_dir="collected_data"):
#     index = []
#     for file in os.listdir(base_dir):
#         path = os.path.join(base_dir, file)
#         if os.path.isfile(path):
#             ext = os.path.splitext(file)[-1].lower()
#             category = CATEGORY_MAP.get(ext, "others")
#             category_dir = os.path.join(base_dir, category)
#             os.makedirs(category_dir, exist_ok=True)
#             new_path = os.path.join(category_dir, file)
#             shutil.move(path, new_path)
#             index.append({"filename": file, "type": ext, "category": category})

#     # Save classification index
#     with open(os.path.join(base_dir, "../collected_data/final_data_index.csv"), "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["filename", "type", "category"])
#         writer.writeheader()
#         writer.writerows(index)
#         print("[+] Saved final_data_index.csv")

def categorize_files(base_dir="collected_data"):
    index = []

    meta_path = os.path.join(base_dir, "temp_meta.jsonl")
    meta_map = {}

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    meta_map[entry['filename']] = entry
                except:
                    continue

    for file in os.listdir(base_dir):
        path = os.path.join(base_dir, file)
        if os.path.isfile(path) and not file.endswith(".jsonl"):
            ext = os.path.splitext(file)[-1].lower()
            category = CATEGORY_MAP.get(ext, "others")
            category_dir = os.path.join(base_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            new_path = os.path.join(category_dir, file)
            shutil.move(path, new_path)

            meta = meta_map.get(file, {
                "filename": file,
                "source": "Unknown",
                "file_type": ext.upper().replace('.', ''),
                "year": "Unknown",
                "topic_keyword": file.split(".")[0],
                "url": "Unknown",
                "scrape_status": "❌"
            })
            meta["category"] = category
            index.append(meta)

    final_csv = os.path.join(base_dir, "final_data_index.csv")
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "source", "file_type", "year",
            "topic_keyword", "url", "scrape_status", "category"
        ])
        writer.writeheader()
        writer.writerows(index)
        print("[+] Saved final_data_index.csv")
