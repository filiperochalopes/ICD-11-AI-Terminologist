import os
from qdrant_client import QdrantClient
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv


# 1. Carrega o token do ambiente
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# 2. Configurações
REPO_ID = "filipelopesmedbr/cid11-agent-mistral-8b"
FILENAME = "ggml-icd11-8b-V2-q4_k.gguf"

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


def load_qdrant_client() -> QdrantClient:
    """
    Cria e retorna um cliente conectado ao Qdrant.
    Exemplo sem autenticação:
    """
    # Se tiver QDRANT_API_KEY ou QDRANT_URL no .env, use:
    host = os.getenv("QDRANT_HOST", "qdrant.filipelopes.me")
    port = os.getenv("QDRANT_PORT", "80")
    return QdrantClient(host=host, port=int(port), prefer_grpc=False)

if __name__ == "__main__":
    model = load_model()
    print(model.n_ctx, model.n_threads, model.n_batch)