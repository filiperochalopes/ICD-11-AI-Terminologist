from V3.classes import GraphState, GraphStateManager
from V3.tools import exact_match_stem_code_tool

def step_002_exact_match_stem_code(state: GraphState) -> GraphState:
    result = exact_match_stem_code_tool._run(state)
    return GraphStateManager(result).update(
        {
            "task_memory": [
                {
                    "name": "step",
                    "content": "step_002_exact_match_stem_code",
                }
            ]
        }
    )