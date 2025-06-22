from V3.classes import GraphState, GraphStateManager
from V3.tools import llm_code_selector

def step_003_llm_select_stem_code(state: GraphState) -> GraphState:
    result = llm_code_selector._run(state)
    return GraphStateManager(result).update(
        {
            "task_memory": [
                {
                    "name": "step",
                    "content": "step_003_llm_select_stem_code",
                }
            ]
        }
    )