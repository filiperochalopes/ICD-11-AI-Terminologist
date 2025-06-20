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

            # Debug print to verify token lists
            print(
                f"Comparing concept tokens {concept_tokens} to FSN tokens {fsn_tokens} and label tokens {label_tokens} for code {hit.get('code')}"
            )

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
