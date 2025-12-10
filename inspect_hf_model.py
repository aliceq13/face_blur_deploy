from huggingface_hub import list_repo_files, hf_hub_download
import os

repo_id = "minchul/cvlface_adaface_vit_base_kprpe_webface12m"
print(f"--- Files in {repo_id} ---")
try:
    files = list_repo_files(repo_id)
    for f in files:
        print(f" - {f}")
except Exception as e:
    print(f"Error listing files: {e}")

print("\n--- Content of wrapper.py ---")
try:
    path = hf_hub_download(repo_id, "wrapper.py")
    with open(path, 'r') as f:
        print(f.read())
except Exception as e:
    print(f"Error reading wrapper.py: {e}")
