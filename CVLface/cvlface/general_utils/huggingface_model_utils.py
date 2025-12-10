import sys
from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import os

# helpfer function to download huggingface repo and use model
def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)


# helpfer function to download huggingface repo and use model
def load_model_from_local_path(path, HF_TOKEN=None):
    # 절대 경로로 변환
    abs_path = os.path.abspath(path)
    wrapper_path = os.path.join(abs_path, 'wrapper.py')
    
    # wrapper.py가 존재하면 상대 경로 문제 해결을 위해 패치
    if os.path.exists(wrapper_path):
        try:
            with open(wrapper_path, 'r') as f:
                content = f.read()
            
            # Windows 경로 구분자 호환성을 위해 replace('\\', '/') 사용
            clean_abs_path = abs_path.replace('\\', '/')
            
            print(f"Patching wrapper.py at {wrapper_path} to use absolute paths...")
            
            # 원본 문자열 타겟
            target_yaml = "'pretrained_model/model.yaml'"
            target_pt = "'pretrained_model/model.pt'"
            
            # 이전의 잘못된 패치 문자열 타겟 (directory relative)
            wrong_yaml = "os.path.join(os.path.dirname(__file__), 'pretrained_model/model.yaml')"
            wrong_pt = "os.path.join(os.path.dirname(__file__), 'pretrained_model/model.pt')"
            
            # 모든 가능한 패턴을 절대 경로로 치환
            content = content.replace(target_yaml, f"'{clean_abs_path}/pretrained_model/model.yaml'")
            content = content.replace(target_pt, f"'{clean_abs_path}/pretrained_model/model.pt'")
            content = content.replace(wrong_yaml, f"'{clean_abs_path}/pretrained_model/model.yaml'")
            content = content.replace(wrong_pt, f"'{clean_abs_path}/pretrained_model/model.pt'")
            
            with open(wrapper_path, 'w') as f:
                f.write(content)
        except Exception as e:
            print(f"Warning: Failed to patch wrapper.py: {e}")

    # 현재 작업 디렉토리 저장
    cwd = os.getcwd()
    
    # 모델 경로로 이동 (여전히 필요할 수 있음)
    os.chdir(abs_path)
    sys.path.insert(0, abs_path)
    
    try:
        # local_files_only=True로 캐시 사용 방지 및 패치된 wrapper.py 사용
        model = AutoModel.from_pretrained(
            abs_path, 
            trust_remote_code=True, 
            token=HF_TOKEN,
            local_files_only=True
        )
    finally:
        # 원래 작업 디렉토리로 복귀
        os.chdir(cwd)
        sys.path.pop(0)
    
    return model


# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)
