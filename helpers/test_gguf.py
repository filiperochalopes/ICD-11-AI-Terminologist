# test_gguf.py -------------------------------------------------
# ! pip install --upgrade "llama-cpp-python==0.2.57"   # já vem com as libs nativas

from pathlib import Path
import pytest

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - handled by test skip
    Llama = None

MODEL_FILE = Path("models/ggml-icd11-8b-V2-q4_k.gguf")

if Llama is not None:
    assert MODEL_FILE.exists(), f"Modelo {MODEL_FILE} não encontrado"

    print("📥 Loading quantised GGUF …")
    llm = Llama(
        model_path=str(MODEL_FILE),
        n_threads=8,       # todos os seus núcleos
        n_batch=32,        # batches menores consomem menos RAM
        n_ctx=1536,        # contexto razoável p/ 10 GB
        use_mmap=True,
        use_mlock=False,   # põe True se seu usuário tiver permissão
    )
    print("✅ Model ready.\n")
else:
    llm = None

def query(concept: str, max_tokens: int = 1536, temp: float = 0.2):
    if llm is None:
        raise RuntimeError("llama_cpp is required to run this test")

    prompt = (
        "<s>[INST] Map the clinical concept to its ICD-11 code. "
        "Add extensions or cluster codes if needed: "
        f"{concept} [/INST]"
    )
    out = llm(prompt, max_tokens=max_tokens, temperature=temp)
    # `out["choices"][0]["text"]` já contém somente a continuação
    response = out["choices"][0]["text"].strip()
    return prompt, response


def test_query_produces_output():
    """Ensure the model returns some text for a sample concept."""
    pytest.importorskip("llama_cpp")
    _, response = query("Carcinoma in Situ of Skin of Perineum")
    assert response

if __name__ == "__main__":
    concept = "Carcinoma in Situ of Skin of Perineum"
    prompt, answer = query(concept)
    print("📨 Prompt:", prompt)
    print("🤖 Answer:", answer)