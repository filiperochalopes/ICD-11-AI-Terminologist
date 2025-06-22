import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

REPO_ID = "filipelopesmedbr/icd11-llm-ministral-8b"
FILENAME = "ggml-icd11-8b-V2-q4_k.gguf"
LOCAL_DIR = "./models"


def ensure_model() -> str:
    os.makedirs(LOCAL_DIR, exist_ok=True)
    target = os.path.join(LOCAL_DIR, FILENAME)
    if os.path.exists(target):
        print(f"[✔] File already exists: {FILENAME}")
        return target

    print(f"[↓] Downloading: {FILENAME} from {REPO_ID}")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=HF_TOKEN,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"[✓] Downloaded: {FILENAME}")
    return path


if __name__ == "__main__":
    ensure_model()
