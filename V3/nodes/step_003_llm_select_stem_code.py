from V3.classes import GraphState, GraphStateManager
from V3.tools import icd11_tool

def step_003_llm_select_stem_code(state: GraphState) -> GraphState:
    result = icd11_tool._run(state)
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