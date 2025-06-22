from V3.classes import GraphState, GraphStateManager


def final_looper(state: GraphState) -> GraphState:
    """
    Checks if we have 3 results or at least as many as the total number of stem codes suggested.
    If so, finalizes the task; otherwise, loops again with the new blacklist.
    """
    sm = GraphStateManager(state)

    # Accumulate previous final codes + the current one
    final_codes = state.final_codes.copy()
    if state.final_code:
        final_codes.append(state.final_code)

    # Check how many stem hits we had
    stem_hits = [hit for m in state.task_memory if m.name == "stem_hits" for hit in m.content]

    print(f"📊 Stem hits: {stem_hits}")
    print(f"📌 Current final_codes: {final_codes}")

    # Check stopping condition
    if len(final_codes) >= 3 or len(final_codes) >= len(stem_hits):
        final_codes_str = "\n".join(f"<line>{code}</line>" for code in final_codes)
        return sm.update(
            {
                "messages": [
                    {
                        "type": "ai",
                        "content": f"""[Final Looper]

Final codes: 

<output>
{final_codes_str}
</output>"""
                    }
                ],
                "final_codes": state.final_code,
                "final_code": "",
                "partial_output_code": "",
            }
        )
    else:
        sm.clear_steps()
        return sm.update(
            {
                "messages": [
                    {
                        "type": "ai",
                        "content": "[Final Looper]\nRestarting the process with the next stem code",
                    }
                ],
                "final_codes": state.final_code,
                "final_code": "",
                "partial_output_code": "",
            }
        )