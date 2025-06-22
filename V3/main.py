from langgraph.graph import END
from V3.nodes import (
    analysis_step_specificity_check,
    step_001_retrieval_stem_codes,
    step_002_exact_match_stem_code,
    step_003_llm_select_stem_code,
    final_looper,
)

from V3.env import builder
from V3.classes import GraphState
from pprint import pprint

def has_task_memory_step(task_memory, name: str, content: str) -> bool:
    """
    Checks if task_memory contains at least one item where both name and content match the given values.

    Args:
        task_memory (List): The state.task_memory list.
        name (str): The name to match (e.g., "step").
        content (str): The content to match (e.g., "step_002_exact_match_stem_code").

    Returns:
        bool: True if a match exists, False otherwise.
    """
    return any(m.name == name and m.content == content for m in task_memory)

def decide_next_after_specificity(state: GraphState) -> str:
    """Return ``end`` if a final code exists, otherwise ``continue``."""
    # Verificando se existe algum código na blacklist, significa que já passou pela comparação inicial
    if any(m.name == "step" and m.content == "step_003_llm_select_stem_code" for m in state.task_memory):
        if state.final_code:
            return "end"
        elif state.partial_output_code:
            return "repeat"
        else:
            return "restart"
    
    if any(m.name == "step" and m.content == "step_002_exact_match_stem_code" for m in state.task_memory):
        if state.partial_output_code:
            return "end"
        else:
            return "resume_llm"
    
    return "end"

def looper(state: GraphState) -> str:
    print("Final codes:")
    pprint(state.final_codes)
    stem_hits = [item for m in state.task_memory if m.name == "stem_hits" for item in m.content]
    print("Condition:", len(state.final_codes) == 3 or len(state.final_codes) == len(stem_hits))
    print("len(state.final_codes):", len(state.final_codes))
    print("len(stem_hits):", len(stem_hits))
    print("stem_hits", stem_hits)
    if len(state.final_codes) == 3 or len(state.final_codes) == len(stem_hits):
        return "end"
    else:
        return "restart"

# Add nodes
builder.add_node("retrieve stem codes", step_001_retrieval_stem_codes)
builder.add_node("compare exact stem concept match", step_002_exact_match_stem_code)
builder.add_node("llm select stem code", step_003_llm_select_stem_code)
builder.add_node("check concept specificity", analysis_step_specificity_check)
builder.add_node("final looper", final_looper)

# Entry point
builder.set_entry_point("retrieve stem codes")

# Sempre vá de retrieval → compare
builder.add_edge("retrieve stem codes", "compare exact stem concept match")
builder.add_edge("compare exact stem concept match", "check concept specificity")

builder.add_conditional_edges(
    "check concept specificity",
    decide_next_after_specificity,
    {
        "end": "final looper",
        "resume_llm": "llm select stem code",
        "restart": "retrieve stem codes",
        "repeat": "check concept specificity",
    },
)

builder.add_edge("llm select stem code", "check concept specificity")

builder.add_conditional_edges(
    "final looper",
    looper,
    {
        "end": END,
        "restart": "retrieve stem codes",
    },
)

builder.set_finish_point("final looper")

# Compile graph
graph = builder.compile()
