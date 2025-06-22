# Imports for type hints and Qdrant client models
from typing import ClassVar, List, Set
from qdrant_client.http.models import Filter
from langchain.tools import BaseTool
from V3.env import qdrant_client, collections, TOP_K
from V3.classes import GraphState, GraphStateManager


class VectorDatabaseRetrieveStemCodes(BaseTool):
    """
    Tool to perform semantic search for the top-K stem leaf codes stored in Qdrant.
    Returns formatted FSN and ICD-11 codes for a given clinical concept text.
    """

    name: ClassVar[str] = "semantic_search_stem_codes"
    description: ClassVar[str] = (
        "Given a clinical-concept text, returns the top-K stem leaf FSNs and ICD-11 codes "
        "from Qdrant, formatted as a text block."
    )

    def _run(self, state: GraphState) -> GraphState:
        sm = GraphStateManager(state)
        # Captura todos os códigos em state.task_memory cujo name é 'blacklist_code'
        blacklist_codes = set()
        print("********* ITEM *********")
        for item in state.task_memory:
            print("item:", item)
            if item.name == "blacklist_code":
                blacklist_codes.add(item.content)
                    
        
        print("++++++++++++++++++++")
        print("blacklist_codes:", blacklist_codes)
        print("++++++++++++++++++++")

        """
        Synchronous execution entry point.
        - Embeds the query text.
        - Searches each Qdrant collection for matching stem leaf codes.
        - Deduplicates results and fetches FSN for each code.
        - Formats and returns the results as a text block.
        """
        # Track seen codes to avoid duplicates
        seen_codes: Set[str] = set()
        # Store formatted results
        results: List[str] = []
        # Store last-stem hits in memory
        stem_hits: List[dict] = []

        # Iterate through each vector collection
        for collection in collections:
            # Embed the query text into a vector
            q_vector = collection.embed(state.clinical_concept_input)

            # Query Qdrant for stem leaf codes using a filter
            hits = qdrant_client.query_points(
                collection_name=collection.name,
                query=q_vector,
                with_payload=True,
                limit=TOP_K,
                query_filter=Filter(
                    must=[
                        {"key": "code_type", "match": {"value": "stem"}},
                        {"key": "is_leaf", "match": {"value": True}},
                    ]
                ),
            )

            # Subtrai os valores que os códigos estão em blacklist
            hits.points = [
                point
                for point in hits.points
                if point.payload["code"] not in blacklist_codes
            ]

            print(">>>>>>>>>>>")
            print("hits.points", [f"{p.payload['code']} - {p.payload['concept_name']}" for p in hits.points])
            print(">>>>>>>>>>>")

            # Process each returned point
            for point in hits.points:
                payload = point.payload or {}
                code = payload.get("code", "").strip()
                label = payload.get("concept_name", "").strip()
                # Skip duplicate or empty codes
                if not code or code in seen_codes:
                    continue

                # Mark code as seen
                seen_codes.add(code)
                # Fetch the fully specified name (FSN) for this code
                fsn = self._fetch_fsn(collection.name, code)

                # Store the code, label, and FSN in memory
                stem_hits.append({**payload, "fsn": fsn, "label": label})

                # Format result: include synonym if FSN differs from label
                if fsn == label:
                    results.append(f"- {code} ({fsn})")
                else:
                    results.append(f"- {code} ({fsn} – synonym: {label})")

        # Prepare and return the output block
        if results:
            header = "Relevant matched stem codes found:"
            # Só adiciona stem_hits a task_memory se ainda não existe
            task_memory: List[dict] = []
            if len([h for h in state.task_memory if h.name == "stem_hits"]) == 0:
                task_memory.append({"name": "stem_hits", "content": stem_hits})
            return sm.update(
                {
                    "task_memory": task_memory,
                    "messages": [
                        {
                            "type": "ai",
                            "content": "\n".join(["[Stem Codes]", header] + results),
                        }
                    ],
                    "context": [
                        {
                            "name": "stem_hits",
                            "content": "\n".join([header] + results),
                        }
                    ],
                }
            )
        # Return fallback message if no matches
        return sm.update(
            {"messages": [{"type": "ai", "content": "No relevant stem codes found."}]}
        )

    def _arun(self, state: GraphState) -> str:
        """
        Asynchronous entry point mapping to the synchronous implementation.
        """
        return self._run(state)

    def _fetch_fsn(self, collection_name: str, code: str) -> str:
        """
        Helper to fetch the FSN for a given code from Qdrant.
        - Applies a filter to retrieve only FSN entries.
        """
        # Query Qdrant for the FSN name_type payload
        response = qdrant_client.query_points(
            collection_name=collection_name,
            with_payload=True,
            limit=1,
            query_filter=Filter(
                must=[
                    {"key": "code", "match": {"value": code}},
                    {"key": "name_type", "match": {"value": "fsn"}},
                ]
            ),
        )
        fsn_payload = response.points[0].payload or {}
        # Extract and return the FSN concept name
        return fsn_payload.get("concept_name", "").strip()
