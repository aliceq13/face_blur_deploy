
import os
import torch
import cv2
import numpy as np
import sys
# from django.conf import settings

# Setup Django settings manually if needed, or just paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'apps', 'videos', 'weights')
MODEL_PATH = os.path.join(WEIGHTS_DIR, 'adaface_vit_base_kprpe_webface12m.pt')

# Add apps path
sys.path.append(os.path.join(BASE_DIR, 'apps', 'videos'))

def check_model():
    print(f"Checking model at: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model file not found!")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        print("[OK] Checkpoint loaded.")
        
        state_dict = checkpoint.get('state_dict', checkpoint)
        keys = list(state_dict.keys())
        print(f"Total keys in state_dict: {len(keys)}")
        print(f"Sample keys: {keys[:5]}")
        
        # Load Architecture
        from apps.videos.adaface.vit import vit_base
        model = vit_base(num_classes=0)
        model.to(device)
        
        # Adjust keys
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            new_state_dict[key] = v
            
        # Load
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        if missing:
            print(f"First 5 missing: {missing[:5]}")
        print(f"Unexpected keys: {len(unexpected)}")
        if unexpected:
            print(f"First 5 unexpected: {unexpected[:5]}")
            
        model.eval()
        
        # Dummy Inference
        dummy_input = torch.randn(1, 3, 112, 112).to(device)
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean().item():.4f}, std: {output.std().item():.4f}")
        
        # Check normalization
        norm = torch.norm(output, p=2, dim=1)
        print(f"Output Norm: {norm.item():.4f}")
        
        # Test similarity with itself (should be 1.0)
        emb = output.cpu().numpy()[0]
        emb = emb / np.linalg.norm(emb)
        sim = np.dot(emb, emb)
        print(f"Self-similarity: {sim:.4f}")

    except Exception as e:
        print(f"[ERROR] Error during check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model()
