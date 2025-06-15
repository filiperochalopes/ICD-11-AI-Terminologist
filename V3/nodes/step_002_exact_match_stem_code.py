from V3.classes import GraphState
from V3.tools import exact_match_stem_code_tool

def step_002_exact_match_stem_code(state: GraphState) -> GraphState:
    result = exact_match_stem_code_tool.run(state)
    return result