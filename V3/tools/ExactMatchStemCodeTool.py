from V3.classes import GraphState, GraphStateManager
from typing import ClassVar
from langchain.tools import BaseTool
import re


class ExactMatchStemCodeTool(BaseTool):
    """
    Returns the code from the last semantic-search hits whose FSN exactly equals
    the input concept. If none match, returns empty string.
    """

    name: ClassVar[str] = "exact_match_stem_code"
    description: ClassVar[str] = (
        "Checks the in-memory hits from semantic search for an exact FSN==concept match."
    )

    def _run(self, state: GraphState) -> GraphState:
        sm = GraphStateManager(state)

        # Normalize input concept: remove punctuation and lowercase
        raw_concept = state.clinical_concept_input
        # Retrieve the list of stem hits stored in state.memory under the
        # "stem_hits" entry. When using Pydantic models, each memory entry is a
        # NamedMemory object, so we access the attributes rather than using
        # dictionary methods.
        stem_hits = []
        for item in state.task_memory:
            if getattr(item, "name", None) == "stem_hits":
                stem_hits = item.content
                break

        blacklist_codes = set(
            item.content
            for item in state.task_memory
            if getattr(item, "name", None) == "blacklist_code"
        )

        # filtrando stem_hits pelo blacklist_codes
        stem_hits = [hit for hit in stem_hits if hit.get("code") not in blacklist_codes]

        if len(stem_hits) == 1:
            # Caso só tenha um stem hit, retorna ele
            return sm.update(
                {
                    "task_memory": [
                        {
                            "name": "blacklist_code",
                            "content": stem_hits[0].get("code"),
                        }
                    ],
                    "partial_output_code": stem_hits[0].get("code"),
                    "messages": [
                        {
                            "type": "ai",
                            "content": f"[Exact Match]\nCode: {stem_hits[0].get('code')}\nFSN: {stem_hits[0].get('fsn')}",
                        }
                    ],
                }
            )

        # Normalize input concept: remove punctuation, lowercase, and split into tokens
        normalized_concept = re.sub(r"[^\w\s]", "", raw_concept.strip()).lower()
        concept_tokens = normalized_concept.split()

        for hit in stem_hits:
            # Normalize FSN and label: remove punctuation, lowercase, split into tokens
            raw_fsn = hit.get("fsn", "")
            raw_label = hit.get("label", "")
            normalized_fsn = re.sub(r"[^\w\s]", "", raw_fsn.strip()).lower()
            normalized_label = re.sub(r"[^\w\s]", "", raw_label.strip()).lower()
            fsn_tokens = normalized_fsn.split()
            label_tokens = normalized_label.split()

            # Compare token sets (order-insensitive) against concept
            if set(fsn_tokens) == set(concept_tokens) or set(label_tokens) == set(
                concept_tokens
            ):
                return sm.update(
                    {
                        "partial_output_code": hit.get("code"),
                        "messages": [
                            {
                                "type": "ai",
                                "content": f"[Exact Match]\nCode: {hit.get('code')}\nFSN: {hit.get('fsn')}\nLabel: {hit.get('label')}",
                            }
                        ],
                    }
                )

        return state

    def _arun(self, state: GraphState) -> GraphState:
        return self._run(state)
