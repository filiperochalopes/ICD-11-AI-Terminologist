from typing import ClassVar
from qdrant_client.http.models import Filter
from V3.env import qdrant_client, collections
from V3.classes import GraphState, GraphStateManager
from V3.tools.LLMBasedTool import LLMBasedTool
import re

# Stop words list to ignore during token comparison
STOP_WORDS = {"and", "of", "the", "a", "an", "in", "on", "with", "without", "due", "to"}


class SpecificityCheckTool(LLMBasedTool):
    name: ClassVar[str] = "specificity_checker"
    description: ClassVar[str] = (
        "Compares a clinical concept with an ICD-11 FSN to decide if the code is more specific, "
        "less specific, or equally specific."
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

        if input_concept_tokens == fsn_tokens:
            # Se o conceito sugerido (FSN) é o mesmo que o conceito clínico de input, então o conceito clínico de input é igual ao FSN |> SAME-AS
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
            # Se o conceito sugerido (FSN) é um superset (ou seja, contém todos os termos do conceito clínico de input) então o conceito clínico de input é mais especifico que o FSN |> NARROWER-THAN
            return sm.update(
                {
                    "task_memory": task_memory,
                    "messages": [
                        {
                            "type": "ai",
                            "content": f"[Specificity Check]\nNARROWER-THAN {code}",
                        }
                    ],
                    "final_code": f"<map_type>NARROWER-THAN</map_type><code>{code}</code>",
                }
            )

        if input_concept_tokens.issuperset(fsn_tokens):
            # Se o conceito clínico de input é um superset (ou seja, contém todos os termos do conceito sugerido (FSN)) então o conceito clínico de input é menos especifico que o FSN |> BROADER-THAN
            return sm.update(
                {
                    "task_memory": task_memory,
                    "messages": [
                        {
                            "type": "ai",
                            "content": f"[Specificity Check]\nBROADER-THAN {code}",
                        }
                    ],
                    "final_code": f"<map_type>BROADER-THAN</map_type><code>{code}</code>",
                }
            )

        # Step 2: LLM judgment if heuristic was inconclusive
        user_msg = f"""You are a medical coding assistant.

Task:
Given this clinical INPUT CONCEPT "{state.clinical_concept_input}" and this FINAL CONCEPT "{fsn}" (which represents the ICD-11 code meaning), compare them and answer:

Is the INPUT CONCEPT more specific, less specific, or equally specific when compared to the FINAL CONCEPT?

Answer strictly with:
- NARROWER-THAN → If the INPUT CONCEPT contains more detail than the FINAL CONCEPT (i.e., INPUT is more specific)
- BROADER-THAN → If the FINAL CONCEPT contains more detail than the INPUT CONCEPT (i.e., FINAL is more specific)
- SAME-AS → If both have the same level of specificity
Do NOT explain. Output only the label.

Assistant:"""

        messages = [("user", user_msg)]
        prompt = self.format_prompt(messages)
        print("📨 Prompt:", prompt)

        response = self.llm_invoke(prompt)
        # Caso a resposta seja NARROWER-THAN, BROADER-THAN ou SAME-AS, então retorna o code correspondente
        if response.content.strip() in ["NARROWER-THAN", "BROADER-THAN", "SAME-AS"]:
            return sm.update(
                {
                    "task_memory": task_memory,
                    "messages": self.convert_llm_response_to_langgraph_messages(
                        response, "Specificity Check"
                    ),
                    "final_code": f"<map_type>{response.content.strip()}</map_type><code>{code}</code>",
                }
            )
        else:
            return sm.update(
                {
                    "messages": [
                        {
                            "type": "ai",
                            "content": "[Specificity Check]\nINCONCLUSIVE. Retrying...",
                        }
                    ],
                    "final_code": "",
                }
            )

    def _arun(self, state: GraphState) -> GraphState:
        return self._run(state)
