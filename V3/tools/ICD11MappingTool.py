from langchain.tools import BaseTool
from langchain_community.chat_models import ChatLlamaCpp
from typing import Dict, Any, ClassVar
from V3.env import GGUF_MODEL_PATH

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
        max_tokens=64, 
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