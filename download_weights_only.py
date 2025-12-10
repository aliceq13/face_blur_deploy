from huggingface_hub import hf_hub_download
import os
import shutil

def download_weights():
    repo_id = 'minchul/cvlface_adaface_vit_base_kprpe_webface12m'
    filename = 'pretrained_model/model.pt'
    
    # Target directory in the project
    target_dir = os.path.join('apps', 'videos', 'weights')
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, 'adaface_vit_base_kprpe_webface12m.pt')
    
    print(f"Downloading {filename} from {repo_id}...")
    
    try:
        # Download the file from Hugging Face
        # We download to a temporary cache first
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir_use_symlinks=False
        )
        
        print(f"Downloaded to: {downloaded_path}")
        
        # Copy to the target location
        print(f"Copying to {target_path}...")
        shutil.copy(downloaded_path, target_path)
        
        print("Success! Model weights are ready.")
        print(f"Path: {target_path}")
        
    except Exception as e:
        print(f"Error downloading weights: {e}")

if __name__ == '__main__':
    download_weights()
