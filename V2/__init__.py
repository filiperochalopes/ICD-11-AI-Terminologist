from langgraph.graph import StateGraph
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatLlamaCpp
from sentence_transformers import SentenceTransformer
from qdrant_client.http.models import Filter

from helpers.model_loader import load_qdrant_client
from helpers.classes import ChatMessage, GraphState
from typing import Dict, Any, ClassVar

# ------------------------------------------------------------------
# 1) Parâmetros e carregamento de recursos globais
# ------------------------------------------------------------------

# Caminho para o arquivo GGUF do seu modelo LoRA fundido/quantizado
GGUF_MODEL_PATH = "models/ggml-icd11-8b-V2-q4_k.gguf"

# Nome da coleção Qdrant já pré-populada com embeddings e payloads
COLLECTION = "icd11_concepts_mpnet"
# Quantos documentos “stem leaf” queremos recuperar do Qdrant
TOP_K = 7

# Instância única do cliente Qdrant (reutilizada em todo o tool)
qdrant_client = load_qdrant_client()

# Instância única do encoder de embeddings SentenceTransformer
# (para CPU, pois toda a aplicação roda em CPU)
embed_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device="cpu"
)

# ------------------------------------------------------------------
# 2) Definição de Tools
# ------------------------------------------------------------------


class QdrantRetrievalTool(BaseTool):
    name: ClassVar[str] = "qdrant_retrieval"
    description: ClassVar[str] = (
        "Dado um conceito clínico em texto puro, retorna um bloco de texto "
        "com as top-K FSNs + códigos ICD-11 recuperados da base Qdrant."
    )
    """
    Tool para recuperar do Qdrant os top-K conceitos 'stem leaf' mais
    próximos do texto de entrada, e formatar um “contexto” concatenado.
    """
    name = "qdrant_retrieval"
    description = (
        "Dado um conceito clínico em texto puro, retorna um bloco de texto "
        "com as top-K FSNs (fully specified names) + códigos ICD-11 recuperados "
        "da coleção Qdrant. Use este contexto para enriquecer a próxima etapa."
    )

    def _run(self, query: str) -> str:
        # 1) Gera embedding para o query recebido
        vector = embed_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        # 2) Consulta a coleção no Qdrant
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

                # Extrai FSN e código dos stem hits e deduplica
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
            # Coleta possíveis leaf_related_codes para pós-coordenação
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
                limit=10,
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
        
        return "\n".join(lines)


    def _arun(self, query: str) -> str:
        # Versão async (não utilizada aqui, mas implementada para compatibilidade)
        return self._run(query)


class ICD11MappingTool(BaseTool):
    name: ClassVar[str] = "icd11_mapping"
    description: ClassVar[str] = (
        "Recebe um dicionário com 'context' e 'concept'. "
        "Retorna string com o código ICD-11 gerado."
    )
    """
    Tool para perguntar ao Llama, dado um contexto (texto vindo do Qdrant)
    e um conceito clínico original, qual o código ICD-11 mais adequado.
    """
    name = "icd11_mapping"
    description = (
        "Recebe um dicionário com chaves 'context' (texto) e 'concept' (texto). "
        "Retorna a string com o código ICD-11 mapeado, possivelmente incluindo "
        "tags de pós-coordenação ou explicações."
    )

    # Instanciaremos um ChatLlamaCpp por classe (pode-se ajustar para reuse global se desejar)
    llm: ClassVar[ChatLlamaCpp] = ChatLlamaCpp(
        model_path=GGUF_MODEL_PATH,
        max_tokens=1536, 
        temperature=0.2,
        n_ctx=1536,
        verbose=True
    )

    def _run(self, args: Dict[str, Any]) -> str:
        # args deve conter 'context' e 'concept'
        context = args.get("context", "")
        concept = args.get("concept", "")

        # Monta o prompt completo incluindo contexto + conceito
        prompt = (
            "<s>[INST] You are a medical coding assistant.\n\n"
            "Instructions:\n"
            "1. Never return a code that begins with “X” (an extension alone).\n"
            "2. If additional detail is needed, combine extension codes (X-prefix) only in association with a valid stem code:\n"
            "   • Use “&” to join a stem code with one or more extensions when the extension adds detail (e.g., “MA14.1&XN109”).\n"
            "   • Use “/” to join two stem codes when both underlying conditions must be represented together (e.g., “DB51/DB30.4”).\n"
            "   • Complex cases may require both “&” and “/” in the same cluster (e.g., “DA63.Z&XT8W/ME24.9Z”).\n"
            "3. Only return the final ICD-11 code (or code cluster) that fully represents the concept—do not include any extra text or explanation.\n\n"
            f"Clinical concept to map: <input>{concept}</input>\n"
            f"{context} [/INST]"
        )

        # Invoca o ChatLlamaCpp para gerar a resposta
        output = self.llm.invoke(prompt)
        return output.content.strip()

    def _arun(self, args: Dict[str, Any]) -> str:
        return self._run(args)


# ------------------------------------------------------------------
# 3) Criação do grafo explícito (LangGraph) com as duas tools
# ------------------------------------------------------------------

# Template inicial para instruir o agente a chamar as duas tools em sequência
initial_prompt = """
You are a medical assistant specialized in mapping clinical concepts to ICD-11 codes.

First, call the qdrant_retrieval tool to obtain the top-K similar 'stem leaf' codes from the ICD-11 database.
Second, call the icd11_mapping tool to produce the final ICD-11 code, passing both:
  - context: the text returned by qdrant_retrieval
  - concept: the user’s original clinical concept description

Always use exactly the keys 'context' and 'concept' when invoking icd11_mapping.
Example:
  {{ "context": "<texto do Qdrant>", "concept": "Example concept" }}
"""

# Initialize tools
qdrant_tool = QdrantRetrievalTool()
icd11_tool = ICD11MappingTool()

# Define node functions to handle state
def retrieval_node(state: GraphState) -> Dict[str, Any]:
    user_message = state.messages[-1].content
    print(f"🔍 User message for retrieval: {user_message}")
    result = qdrant_tool._run(user_message)
    return {
        "context": result,
        "messages": state.messages + [ChatMessage(type="ai", content=f"[Qdrant Results]\n{result}")]
    }

def mapping_node(state: GraphState) -> Dict[str, Any]:
    user_message = state.messages[-1].content
    result = icd11_tool._run({"context": state.context, "concept": user_message})
    return {
        "messages": state.messages + [ChatMessage(type="ai", content=f"[ICD11 Mapping]\n{result}")]
    }

# Create the graph
builder = StateGraph(GraphState)

# Add nodes
builder.add_node("retrieval", retrieval_node)
builder.add_node("mapping", mapping_node)

# Connect nodes and define entry/exit points
builder.set_entry_point("retrieval")
builder.add_edge("retrieval", "mapping")
builder.set_finish_point("mapping")

# Compile the graph
graph = builder.compile()

# Export the graph as app for LangGraph Studio
app = graph