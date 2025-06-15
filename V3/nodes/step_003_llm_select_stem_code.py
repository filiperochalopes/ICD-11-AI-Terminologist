from helpers.classes import GraphState
from V3.tools import icd11_tool

def step_003_llm_select_stem_code(state: GraphState) -> GraphState:
    result = icd11_tool._run(state)
    return result