import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load Hugging Face token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# List of models to download
models_to_download = [
    {"repo": "bartowski/Ministral-8B-Instruct-2410-GGUF", "filename": "Ministral-8B-Instruct-2410-Q4_K_L.gguf"},
    {"repo": "filipelopesmedbr/icd11-llm-ministral-8b", "filename": "ggml-icd11-8b-V2-q4_k.gguf"},
    {"repo": "filipelopesmedbr/icd11-llm-ministral-8b", "filename": "ggml-icd11-8b-q4_k.gguf"},
    {"repo": "TheBloke/deepseek-coder-6.7B-instruct-GGUF", "filename": "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"},
    {"repo": "TheBloke/Llama-2-7B-Chat-GGUF", "filename": "llama-2-7b-chat.Q4_K_M.gguf"},
    {"repo": "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", "filename": "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf"},
]

# Local models directory
LOCAL_DIR = "./models"

def ensure_models():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    for model in models_to_download:
        target_path = os.path.join(LOCAL_DIR, model["filename"])
        if os.path.exists(target_path):
            print(f"[✔] File already exists: {model['filename']}")
            continue

        print(f"[↓] Downloading: {model['filename']} from {model['repo']}")
        hf_hub_download(
            repo_id=model["repo"],
            filename=model["filename"],
            token=HF_TOKEN,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"[✓] Downloaded: {model['filename']}")

if __name__ == "__main__":
    ensure_models()