from V3.tools.SpecificityCheckTool import LLMBasedTool
from typing import ClassVar
from V3.classes import GraphState, GraphStateManager


class LLMCodeSelector(LLMBasedTool):
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

    def _run(self, state: GraphState) -> GraphState:
        sm = GraphStateManager(state)
        # Extract core parameters

        # captura da lista state.context onde name = "stem_hits"
        # Capture the last item in state.context where "hit" is in item.name
        last_hit_item = next(
            (item for item in reversed(state.context) if "hit" in item.name), None
        )
        context = last_hit_item.content if last_hit_item else ""

        # Build dynamic prompt for model
        user_msg = f"""You are a clinical coding assistant.

Task:
Select the single ICD-11 code that best matches the given clinical concept.
- The chosen code must be as specific as possible without exceeding the concept's specificity.
- Do NOT introduce any additional qualifiers or attributes not present in the concept.
- Return ONLY the ICD-11 code with no description or explanation.

Concept: {state.clinical_concept_input}
Context: {context}

Assistant:"""

        messages = [("user", user_msg)]
        prompt = self.format_prompt(messages)

        # Invoke the model to generate the code mapping
        response = self.llm_invoke(prompt)
        # Return the cleaned output
        return sm.update(
            {
                "partial_output_code": response.content.strip(),
                "messages": self.convert_llm_response_to_langgraph_messages(
                    response, "LLM Stem Code Selection"
                ),
            }
        )

    def _arun(self, state: GraphState) -> GraphState:
        """
        Asynchronous execution entry point.
        Delegates to the synchronous _run method.
        """
        return self._run(state)
