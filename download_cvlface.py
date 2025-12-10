from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import os
import torch
import sys
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import numpy as np

# Helper function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        try:
            hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
        except Exception as e:
            print(f"Error downloading files.txt: {e}")
            return

    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    
    # Filter empty strings and add essential files
    target_files = [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']
    # Remove duplicates
    target_files = list(set(target_files))

    for file in target_files:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            print(f"Downloading {file}...")
            try:
                hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
            except Exception as e:
                print(f"Failed to download {file}: {e}")

# Helper function to load model from local path
def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        model = None
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    return model

# Helper function to download and load model by repo_id
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

if __name__ == '__main__':
    # You can set your HF token here if needed, or leave as None for public models
    HF_TOKEN = os.environ.get('HF_TOKEN', None) 
    
    # 1. Load Main Model
    repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface12m'
    path = os.path.expanduser(f'~/.cvlface_cache/{repo_id}')
    print(f"Loading model: {repo_id} to {path}")
    model = load_model_by_repo_id(repo_id, path, HF_TOKEN)
    
    if model is None:
        print("Failed to load model.")
        sys.exit(1)
    
    print("Model loaded successfully.")

    # 2. Create Dummy Input
    print("Creating dummy input...")
    # Create a dummy image (112x112 RGB)
    img = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
    
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    input_tensor = trans(img).unsqueeze(0)  # torch.randn(1, 3, 112, 112)
    
    # 3. Load Aligner Model (KPRPE also takes keypoints locations as input)
    aligner_repo_id = 'minchul/cvlface_DFA_mobilenet'
    aligner_path = os.path.expanduser(f'~/.cvlface_cache/{aligner_repo_id}')
    print(f"Loading aligner: {aligner_repo_id} to {aligner_path}")
    aligner = load_model_by_repo_id(aligner_repo_id, aligner_path, HF_TOKEN)
    
    if aligner:
        print("Aligner loaded. Running alignment...")
        # Note: The aligner might expect specific input format. 
        # Based on user snippet: aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner(input)
        try:
            with torch.no_grad():
                aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner(input_tensor)
            print("Alignment successful.")
            
            keypoints = orig_ldmks
            
            # 4. Run Inference
            print("Running inference...")
            with torch.no_grad():
                out = model(input_tensor, keypoints)
            print("Inference successful!")
            print(f"Output shape: {out.shape if hasattr(out, 'shape') else 'Unknown'}")
            print(out)
            
        except Exception as e:
            print(f"Error during inference: {e}")
    else:
        print("Failed to load aligner.")
