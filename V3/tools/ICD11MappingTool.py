# Module: ICD-11 Mapping Tool
# This tool maps a clinical concept and supporting context text to the most appropriate ICD-11 code
# using a local ChatLlamaCpp model.

from langchain.tools import BaseTool
from langchain_community.chat_models import ChatLlamaCpp
from typing import Dict, Any, ClassVar
from V3.env import GGUF_MODEL_PATH

class ICD11MappingTool(BaseTool):
    """
    LangChain Tool: ICD-11 Mapping
    Selects the ICD-11 code that best matches a given clinical concept.
    Ensures the chosen code's specificity does not exceed that of the input concept.
    """
    name: ClassVar[str] = "icd11_mapping"
    description: ClassVar[str] = (
        "Maps a clinical concept to the most fitting ICD-11 code. "
        "The selected code will be as specific as possible but will not be more specific "
        "than the provided concept."
    )

    # Initialize ChatLlamaCpp once per class for efficiency and reuse
    llm: ClassVar[ChatLlamaCpp] = ChatLlamaCpp(
        model_path=GGUF_MODEL_PATH,  # Path to the local GGUF model file
        max_tokens=64,               # Limit response length
        temperature=0.2,             # Low randomness for consistent outputs
        n_ctx=1536,                  # Context window size for input
        verbose=True                 # Enable model-level logging for debugging
    )

    def _run(self, input_args: Dict[str, Any]) -> str:
        """
        Synchronous execution entry point.
        Args:
            input_args: dict containing:
                - 'concept': str, the clinical concept to code.
                - 'context': str, supporting text (e.g., vector search results).
        Returns:
            str: The selected ICD-11 code (possibly with post-coordination tags).
        """
        # Extract core parameters
        concept = input_args.get("concept", "").strip()
        context = input_args.get("context", "").strip()

        # Build a clear, deterministic prompt
        prompt = (
            "<s>[INST] You are a clinical coding assistant.\n\n"
            "Task:\n"
            "Select the single ICD-11 code that best matches the given clinical concept.\n"
            "- The chosen code must be as specific as possible without exceeding the concept's specificity.\n"
            "- Do NOT introduce any additional qualifiers or attributes not present in the concept.\n\n"
            "Example:\n"
            "  Concept = 'fracture of the femoral head'\n"
            "  Do NOT select 'open fracture of the femoral head'.\n\n"
            f"Concept: {concept}\n"
            f"Context: {context}\n"
            "[/INST]"
        )

        # Invoke the model to generate the code mapping
        response = self.llm.invoke(prompt)
        # Return the cleaned output
        return response.content.strip()

    def _arun(self, input_args: Dict[str, Any]) -> str:
        """
        Asynchronous execution entry point.
        Delegates to the synchronous _run method.
        """
        return self._run(input_args)