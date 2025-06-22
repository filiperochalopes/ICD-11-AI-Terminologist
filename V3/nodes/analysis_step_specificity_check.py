from V3.classes import GraphState, GraphStateManager
from V3.tools import specificity_check_tool

def analysis_step_specificity_check(state: GraphState) -> GraphState:
    result: GraphState = specificity_check_tool._run(state)
    return GraphStateManager(result).update(
        {
            "task_memory": [
                {
                    "name": "step",
                    "content": "analysis_step_specificity_check",
                },
            ]
        }
    )