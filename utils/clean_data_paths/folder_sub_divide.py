import shutil
from pathlib import Path


# configurations
base_folder = Path("../../artifacts/data/grouped_data/")
source_folders = [base_folder / "group_12", base_folder / "group_13"]
group_labels = ['a', 'b', 'c', 'd']
group_size = 16


for source_folder in source_folders:
    pdf_files = sorted([f for f in source_folder.glob("*.pdf")])
    
    if not pdf_files:
        print(f"No PDFs found in {source_folder}")
        continue

    print(f"Processing {len(pdf_files)} PDFs in {source_folder.name}")

    for idx, label in enumerate(group_labels):
        # Construct destination path
        dest_group_name = f"{source_folder.name}_{label}"
        dest_folder =  source_folder / dest_group_name
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Select 4 files per group
        start = idx * group_size
        end = start + group_size
        group_files = pdf_files[start:end]

        for file_path in group_files:
            shutil.move(str(file_path), str(dest_folder / file_path.name))
            print(f"Moved {file_path.name} -> {dest_folder}")
