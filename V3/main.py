from langgraph.graph import END
from V3.nodes import (
    analysis_step_specificity_check,
    step_001_retrieval_stem_codes,
    step_002_exact_match_stem_code,
    step_003_llm_select_stem_code,
)

from V3.env import builder
from V3.classes import GraphState


def decide_next_after_specificity(state: GraphState) -> str:
    """Return ``end`` if a final code exists, otherwise ``continue``."""
    # Verificando se existe algum código na blacklist, significa que já passou pela comparação inicial
    for m in state.task_memory:
        if m.name == "step" and m.content == "step_002_exact_match_stem_code":
            if state.final_code:
                return "end"
            else:
                return "resume_llm"
        elif m.name == "step" and m.content == "step_003_llm_select_stem_code":
            if state.final_code:
                return "end"
            else:
                # start the process again
                return "restart"

    return "end"


# Add nodes
builder.add_node("retrieve stem codes", step_001_retrieval_stem_codes)
builder.add_node("compare exact stem concept match", step_002_exact_match_stem_code)
builder.add_node("llm select stem code", step_003_llm_select_stem_code)
builder.add_node("check concept specificity", analysis_step_specificity_check)

# Entry point
builder.set_entry_point("retrieve stem codes")

# Sempre vá de retrieval → compare
builder.add_edge("retrieve stem codes", "compare exact stem concept match")
builder.add_edge("compare exact stem concept match", "check concept specificity")

builder.add_conditional_edges(
    "check concept specificity",
    decide_next_after_specificity,
    {
        "end": END,
        "resume_llm": "llm select stem code",
        "restart": "retrieve stem codes",
    },
)

builder.add_edge("llm select stem code", "check concept specificity")

# Declare finish points em ambos:
# - Se compare encontrou algo (mensagem não vazia), o grafo para ali.
# - Se compare NÃO encontrou (mensagem vazia), ele sai para mapping, e depois para finish.
builder.set_finish_point("check concept specificity")

# Compile graph
graph = builder.compile()
