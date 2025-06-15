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

        if not state.partial_output_code:
            return sm.update(
                {
                    "messages": [
                        {
                            "type": "ai",
                            "content": "[Specificity Check]\nNo code found. Next step",
                        }
                    ],
                }
            )

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

        print("📨 Prompt:", state.clinical_concept_input)
        print("Código que vai para blacklist:", codes[0])
        print("🤖 Answer:", code)

        # Verifica se esse é o útltimo passo do código, pois nesse caso ele precisará retornar algo de qualquer forma para o final_code
        task_memory = [{"name": "blacklist_code", "content": codes[0]}]
        force_final_code = False
        for m in state.task_memory:
            if m.name == "step" and m.content == "final_step_flag":
                force_final_code = True
                break

        if input_concept_tokens == fsn_tokens:
            return sm.update(
                {
                    "task_memory": task_memory,
                    "messages": [
                        {
                            "type": "ai",
                            "content": f"[Specificity Check]\nSAME-AS {code}",
                        }
                    ],
                    "final_code": f"<map_type>SAME-AS</map_type><code>{code}</code>",
                }
            )
        if fsn_tokens.issuperset(input_concept_tokens):
            rtn_state = sm.update(
                {
                    "task_memory": [{"name": "blacklist_code", "content": codes[0]}],
                    "messages": [
                        {
                            "type": "ai",
                            "content": f"[Specificity Check]\nNARROWER-THAN {code}",
                        }
                    ],
                }
            )
            if force_final_code:
                rtn_state = GraphStateManager(rtn_state).update(
                    {"final_code": f"<map_type>NARROWER-THAN</map_type><code>{code}"}
                )
            return rtn_state
        if input_concept_tokens.issuperset(fsn_tokens):
            rtn_state = sm.update(
                {
                    "task_memory": [{"name": "blacklist_code", "content": codes[0]}],
                    "messages": [
                        {
                            "type": "ai",
                            "content": f"[Specificity Check]\nBROADER-THAN {code}",
                        }
                    ],
                }
            )
            if force_final_code:
                rtn_state = GraphStateManager(rtn_state).update(
                    {"final_code": f"<map_type>BROADER-THAN</map_type><code>{code}"}
                )
            return rtn_state

        # Step 2: LLM judgment if heuristic was inconclusive
        prompt = (
            "<s>[INST] You are a medical coding assistant.\n\n"
            "Task:\n"
            f'Given this clinical INPUT CONCEPT "{state.clinical_concept_input}" and this FINAL CONCEPT "{fsn}" (which represents the ICD-11 code meaning), compare them and answer:\n\n'
            "Is the INPUT CONCEPT more specific, less specific, or equally specific when compared to the FINAL CONCEPT?\n"
            "- Answer strictly with:\n"
            "  - NARROWER-THAN → If the INPUT CONCEPT contains more detail than the FINAL CONCEPT (i.e., INPUT is more specific)\n"
            "  - BROADER-THAN → If the FINAL CONCEPT contains more detail than the INPUT CONCEPT (i.e., FINAL is more specific)\n"
            "  - SAME-AS → If both have the same level of specificity\n"
            "Do NOT explain. Output only the label.\n"
            "[/INST]"
        )

        print("📨 Prompt:", prompt)

        response = self.llm.invoke(prompt)
        rtn_state = sm.update(
            {
                "messages": [
                    {
                        "type": "ai",
                        "content": f"[Specificity Check]\n{response.content.strip()} {code}",
                    }
                ]
            }
        )

        if force_final_code:
            rtn_state = GraphStateManager(rtn_state).update(
                {
                    "final_code": f"<map_type>{response.content.strip()}</map_type><code>{code}"
                }
            )

        return rtn_state

    def _arun(self, state: GraphState) -> GraphState:
        return self._run(state)
