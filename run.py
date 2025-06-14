# run.py

from langgraph.graph import END, StateGraph
from model_loader import load_model, load_qdrant_client
from sentence_transformers import SentenceTransformer

from typing import TypedDict, Optional
from llama_cpp import Llama
from pydantic import BaseModel, Field
from typing import Annotated
from qdrant_client.http.models import Filter

# 1) Estado compartilhado
AgentState = dict

# 2) Carrega Llama + Qdrant
llm: Llama = load_model()
print("🧠 Llama (gguf) loaded:", llm)

qdrant_client = load_qdrant_client()
print("🔍 Qdrant client ready:", qdrant_client)

# --- Parâmetros de busca no Qdrant ---
TOP_K = 7                   # quantos documentos recuperar
COLLECTION = "icd11_concepts_mpnet"  # nome da coleção Qdrant

# ---- Função de embeddings (usando SentenceTransformer) ----
def get_embedding(text: str, model: Optional[SentenceTransformer] = None) -> list[float]:
    """
    Gera embedding usando o modelo `sentence-transformers/all-mpnet-base-v2`.
    """
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.tolist()

# 3) Nó de Retrieval: lê `state["input"]`, faz consulta no Qdrant
class RetrievalInput(BaseModel):
    input: Annotated[str, Field(description="O conceito clínico inicial")]

def retrieve_from_qdrant(state: AgentState) -> tuple[str, AgentState]:
    """
    1) Lê state["input"]
    2) Gera embedding via SentenceTransformer
    3) Consulta TOP_K resultados no Qdrant
    4) Concatena FSNs recuperadas em state["context"]
    """
    user_text = state["input"]

    # 1) Gera embedding para o texto de entrada usando SentenceTransformer
    emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    vector = get_embedding(user_text, emb_model)

    # 2) Consulta Qdrant para stem codes
    stem_hits = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=vector,
        with_payload=True,
        limit=TOP_K,
        query_filter=Filter(
            must=[
                {"key": "code_type", "match": {"value": "stem"}},
                {"key": "is_leaf",   "match": {"value": True}}
            ]
        ),
    )
    print(f"🔍 Qdrant retrieved {len(stem_hits.points)} stem hits for query: {user_text}")

    # Extrai FSN e código dos stem hits e deduplica, coletando leaf_related_codes
    stem_codes = []
    seen_stem_codes = set()
    related_codes = set()
    for hit in stem_hits.points:
        payload = hit.payload or {}
        code = payload.get("code", "").strip()
        title = payload.get("concept_name", "").strip()
        if code and code not in seen_stem_codes:
            seen_stem_codes.add(code)
            stem_codes.append(f"- {code} ({title})")
        for rel in payload.get("leaf_related_codes", []):
            if rel:
                related_codes.add(rel.strip())

    # 3) Busca extensões para leaf_related_codes
    extension_hits_related = []
    if related_codes:
        extension_hits_related = qdrant_client.query_points(
            collection_name=COLLECTION,
            query=vector,
            with_payload=True,
            limit=len(related_codes),
            query_filter=Filter(
                must=[
                    {"key": "code", "match": {"any": list(related_codes)}},
                    {"key": "code_type", "match": {"value": "extension"}}
                ]
            ),
        )
        print(f"🔍 Qdrant retrieved {len(extension_hits_related.points)} related extension hits")

    # 4) Busca gerais de extensões (top 5) para sugestões adicionais
    general_extension_hits = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=vector,
        with_payload=True,
        limit=5,
        query_filter=Filter(
            must=[
                {"key": "code_type", "match": {"value": "extension"}}
            ]
        ),
    )
    print(f"🔍 Qdrant retrieved {len(general_extension_hits.points)} general extension hits")

    # Deduplica e combina extensões de ambas buscas
    seen_ext_codes = set()
    extension_options = []
    def add_extensions(hits_list):
        for hit in hits_list.points:
            payload = hit.payload or {}
            code = payload.get("code", "").strip()
            title = payload.get("title", payload.get("concept_name", "")).strip()
            if code and code not in seen_ext_codes:
                seen_ext_codes.add(code)
                extension_options.append(f"- {code} ({title})")

    add_extensions(extension_hits_related)
    add_extensions(general_extension_hits)

    # Monta a string de contexto final
    lines = []
    if stem_codes:
        lines.append("Relevant matched stem codes found:")
        lines.extend(stem_codes)
    else:
        lines.append("Relevant matched stem codes found: None.")

    if extension_options:
        lines.append("\nPost-coordination options:")
        lines.extend(extension_options)

    context_text = "\n".join(lines)

    state["context"] = context_text
    return "got_context", state


# 4) Nó do Llama: consome `state["input"]` + `state["context"]`
class LlamaInput(BaseModel):
    input:   Annotated[str, Field(description="Conceito original")]
    context: Annotated[str, Field(description="Contexto recuperado no Qdrant")]

def generate_response(state: AgentState) -> tuple[str, AgentState]:
    """
    Monta prompt usando contexto + input e chama o Llama para gerar resposta.
    """
    last_input    = state["input"]
    retrieved_ctx = state.get("context", "")

    # Se por algum motivo não houver contexto, incluímos nota simples:
    if not retrieved_ctx:
        prefix = ""
    else:
        prefix = retrieved_ctx + "\n\n"

    prompt = (
        f"<s>[INST] Based on this data:\n"
        f"{prefix}"
        f"Map the clinical concept to its ICD-11 code:\n"
        f"{last_input} [/INST]"
    )

    output = llm(prompt, max_tokens=200, temperature=0.2)
    result_text = output["choices"][0]["text"].strip()
    print("🤖 Llama Response:", result_text)

    # Coloca a saída em state["output"]
    state["output"] = result_text
    return END, state


# 5) Montagem do grafo LangGraph
builder = StateGraph(AgentState)

# ► Nó que faz retrieval no Qdrant ("retrieval_from_qdrant")
builder.add_node("retrieval_from_qdrant", retrieve_from_qdrant)

# ► Nó que chama o Llama ("llama_reply")
builder.add_node("llama_reply", generate_response)

# Definimos a ordem:
builder.set_entry_point("retrieval_from_qdrant")
builder.add_edge("retrieval_from_qdrant", "llama_reply")
builder.add_edge("llama_reply", END)

# Compila o app LangGraph
app = builder.compile()


# 6) Execução em modo CLI
if __name__ == "__main__":
    print("📥 Type a concept to map to ICD-11 (type 'exit' to quit)")
    while True:
        user_input = input("🧑 Concept: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        state0 = {"input": user_input}
        # Essa chamada retorna uma tupla (END, novo_estado)
        _, new_state = app.invoke(state0)

        # Mostra o contexto que veio do Qdrant
        print("🔍 Qdrant Context:\n", new_state.get("context", ""))
        # Mostra a resposta final do Llama
        print("🤖 Mapped ICD-11:", new_state.get("output", ""))
        print("---")