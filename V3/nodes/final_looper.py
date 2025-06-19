from V3.classes import GraphState, GraphStateManager
from V3.tools import exact_match_stem_code_tool


def final_looper(state: GraphState) -> GraphState:
    """
    Verifica se temos 3 resultados ou pelo menos a quantidade total de stem codes sugeridos, se sim, finaliza a task, senão reinicia com a nova blacklist
    """

    no_stem_results = 
    for m in state.task_memory:
        if m.name ==

    result = exact_match_stem_code_tool._run(state)
    return GraphStateManager(result).update(
        {
            "messages": [
                {
                    "type": "ai",
                    "content": "[Final Checker]\nfinal_looper",
                }
            ]
        }
    )
    return GraphStateManager(result).update(
        {
            "messages": [
                {
                    "type": "ai",
                    "content": "[Final Results]\nfinal_looper",
                }
            ]
        }
    )