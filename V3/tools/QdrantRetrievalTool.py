from langchain.tools import BaseTool
from typing import ClassVar
from qdrant_client.http.models import Filter
from V3.env import qdrant_client, embed_model, COLLECTION, TOP_K

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