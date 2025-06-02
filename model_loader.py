import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# 1. Carrega o token do ambiente
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Configurações
REPO_ID = "filipelopesmedbr/cid11-agent-mistral-8b"
FILENAME = "ggml-icd11-8b-q4_k.gguf"

# 3. Baixa o arquivo GGUF apenas se não estiver no cache
def get_model_path() -> str:
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        token=HF_TOKEN,
        local_dir="./models",
        local_dir_use_symlinks=False,
    )

# 4. Carrega o modelo com configuração otimizada para CPU
def load_model() -> Llama:
    model_path = get_model_path()
    return Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=8,
        n_batch=64,
        use_mlock=True,
        use_mmap=True,
        verbose=False,
    )

if __name__ == "__main__":
    model = load_model()
    print(model.n_ctx, model.n_threads, model.n_batch)