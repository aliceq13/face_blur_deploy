
import torch
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from apps.videos.adaface_wrapper import AdaFaceWrapper

def check_dim():
    base_dir = os.path.join(os.getcwd(), 'apps', 'videos')
    weight_path = os.path.join(base_dir, 'weights', 'adaface_vit_base_kprpe_webface4m.pt')
    
    if not os.path.exists(weight_path):
        print(f"Weight file not found at {weight_path}")
        # Fallback to checking architecture default
        print("Checking architecture default without weights...")
        wrapper = AdaFaceWrapper("dummy_path", device='cpu', architecture='vit')
    else:
        print(f"Loading weights from {weight_path}")
        wrapper = AdaFaceWrapper(weight_path, device='cpu', architecture='vit')
        
    dummy_input = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        output = wrapper.model(dummy_input)
        
    print(f"Output shape: {output.shape}")
    print(f"Embedding dimension: {output.shape[1]}")

if __name__ == "__main__":
    check_dim()
