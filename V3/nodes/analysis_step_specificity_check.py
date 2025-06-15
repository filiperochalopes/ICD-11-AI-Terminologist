from V3.classes import GraphState
from V3.tools import specificity_check_tool

def analysis_step_specificity_check(state: GraphState) -> GraphState:
    result = specificity_check_tool._run(state)
    return result