from langgraph.graph import END
from V3.nodes import (
    analysis_step_specificity_check,
    step_001_retrieval_stem_codes,
    step_002_exact_match_stem_code,
    step_003_llm_select_stem_code,
)

from V3.env import builder
from V3.classes import GraphState


def process_done(state: GraphState) -> str:
    """
    Determines next step after exact match check:
    - If a code was found (non-empty message), finish the graph.
    - Else, proceed to LLM mapping.
    """
    if state.final_code:
        return "end"
    return "continue"


# Add nodes
builder.add_node("retrieve stem codes", step_001_retrieval_stem_codes)
builder.add_node("compare exact stem concept match", step_002_exact_match_stem_code)
builder.add_node("llm select stem code", step_003_llm_select_stem_code)
builder.add_node(
    "check concept specificity", analysis_step_specificity_check
)

# Entry point
builder.set_entry_point("retrieve stem codes")

# Sempre vá de retrieval → compare
builder.add_edge("retrieve stem codes", "compare exact stem concept match")
builder.add_edge("compare exact stem concept match", "check concept specificity")

builder.add_conditional_edges(
    "check concept specificity", process_done, {"end": END, "continue": "llm select stem code"}
)

builder.add_edge("llm select stem code", "check concept specificity")

builder.add_conditional_edges(
    "check concept specificity", process_done, {"end": END, "continue": "retrieve stem codes"}
)

# Declare finish points em ambos:
# - Se compare encontrou algo (mensagem não vazia), o grafo para ali.
# - Se compare NÃO encontrou (mensagem vazia), ele sai para mapping, e depois para finish.
builder.set_finish_point("check concept specificity")

# Compile graph
graph = builder.compile()
