# File: V3/tools/SpecificityCheckTool.py
from typing import ClassVar, Dict
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatLlamaCpp
from qdrant_client.http.models import Filter
from V3.env import GGUF_MODEL_PATH, qdrant_client, collections
from V3.classes import GraphState, GraphStateManager
import re

# Stop words list to ignore during token comparison
STOP_WORDS = {"and", "of", "the", "a", "an", "in", "on", "with", "without"}


class SpecificityCheckTool(BaseTool):
    """
    LangChain Tool: Specificity Check
    Step 1: Performs token-set comparison between the input concept and ICD-11 FSN.
    Step 2: If inconclusive, asks a local LLM for judgment: more_specific / less_specific / same_specificity.
    """

    name: ClassVar[str] = "specificity_checker"
    description: ClassVar[str] = (
        "Compares a clinical concept with an ICD-11 FSN to decide if the code is more specific, "
        "less specific, or equally specific."
    )

    # LLM setup
    llm: ClassVar[ChatLlamaCpp] = ChatLlamaCpp(
        model_path=GGUF_MODEL_PATH,
        max_tokens=64,
        temperature=0.2,
        n_ctx=1536,
        verbose=True,
    )

    def _run(self, state: GraphState) -> GraphState:
        sm = GraphStateManager(state)

        # Captura o código parcial
        code = state.partial_output_code
        # Separa o código pelos caracteres com regex: & ou /
        codes = re.split(r"[&/]", code)
        # captura os códigos na busca do banco vetorizado
        results = qdrant_client.query_points(
            collection_name=collections[0].name,
            limit=len(codes),
            with_payload=True,
            query_filter=Filter(
                must=[
                    {"key": "code", "match": {"any": codes}},
                    {"key": "name_type", "match": {"value": "fsn"}},
                ]
            ),
        )

        # Extrai os concept_name e une em uma única string
        fsn = " ".join([result.payload["concept_name"] for result in results.points])

        # Step 1: Token-set heuristic
        def normalize(text: str) -> set[str]:
            txt = re.sub(r"[^\w\s]", "", text.lower())
            return {t for t in txt.split() if t not in STOP_WORDS}

        input_concept_tokens = normalize(state.clinical_concept_input)
        fsn_tokens = normalize(fsn)

        if input_concept_tokens == fsn_tokens:
            return sm.update(
                {
                    "messages": [
                        {
                            "role": "ai",
                            "content": f"SAME-AS {code}",
                        }
                    ],
                    "final_code": f"<map_type>SAME-AS</map_type><code>{code}</code>",
                }
            )
        if fsn_tokens.issuperset(input_concept_tokens):
            return sm.update(
                {
                    "messages": [
                        {
                            "role": "ai",
                            "content": f"NARROWER-THAN {code}",
                        }
                    ]
                }
            )
        if input_concept_tokens.issuperset(fsn_tokens):
            return sm.update(
                {
                    "messages": [
                        {
                            "role": "ai",
                            "content": f"BROADER-THAN {state.partial_output_code}",
                        }
                    ]
                }
            )

        # Step 2: LLM judgment if heuristic was inconclusive
        prompt = (
            "<s>[INST] You are a medical coding assistant.\n\n"
            "Task:\n"
            f"Given this clinical concept: '{state.clinical_concept_input}'\n"
            f"And this ICD-11 FSN: '{fsn}'\n\n"
            "Question:\n"
            "Does the ICD-11 code add details that are not present in the concept?\n"
            "- Answer strictly with one of: 'NARROWER-THAN' for more_specific, 'BROADER-THAN' for less_specific, 'SAME-AS' for same specificity.\n"
            "- Do NOT explain. Return only the single term.\n"
            "[/INST]"
        )

        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _arun(self, state: GraphState) -> GraphState:
        return self._run(state)
