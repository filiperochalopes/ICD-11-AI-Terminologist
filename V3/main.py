from langgraph.graph import END
from V3.nodes.step_003_llm_select_stem_code import step_003_llm_select_stem_code
from V3.nodes.step_001_retrieval_stem_codes import step_001_retrieval_stem_codes 
from V3.nodes.step_002_exact_match_stem_code import step_002_exact_match_stem_code

from V3.env import builder

def check_exact_match(state):
    """
    Determines next step after exact match check:
    - If a code was found (non-empty message), finish the graph.
    - Else, proceed to LLM mapping.
    """
    if state.messages[-1].content.strip():
        return "end"
    return "continue"

# Add nodes
builder.add_node("retrieval", step_001_retrieval_stem_codes)
builder.add_node("compare", step_002_exact_match_stem_code)
builder.add_node("mapping", step_003_llm_select_stem_code)

# Entry point
builder.set_entry_point("retrieval")

# Sempre vá de retrieval → compare
builder.add_edge("retrieval", "compare")

builder.add_conditional_edges(
    "compare",
    check_exact_match,
    {
        "end": END,
        "continue": "mapping"
    }
)

# Declare finish points em ambos:
# - Se compare encontrou algo (mensagem não vazia), o grafo para ali.
# - Se compare NÃO encontrou (mensagem vazia), ele sai para mapping, e depois para finish.
builder.set_finish_point("compare")
builder.set_finish_point("mapping")

# Compile graph
graph = builder.compile()