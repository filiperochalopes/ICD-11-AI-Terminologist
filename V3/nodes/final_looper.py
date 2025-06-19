from V3.classes import GraphState, GraphStateManager
from V3.tools import exact_match_stem_code_tool


def final_looper(state: GraphState) -> GraphState:
    """
    Verifica se temos 3 resultados ou pelo menos a quantidade total de stem codes sugeridos, se sim, finaliza a task, senão reinicia com a nova blacklist
    """

    # forma o array de resultados finas
    final_codes = []
    final_codes.append(state.final_code)
    
    # Verifica a quantidade de stem code que ficou
    print([m.content for m in state.task_memory if m.name == "stem_hits"])

    # Verifica os resultados que já temos
    

    result = exact_match_stem_code_tool._run(state)
    return state
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