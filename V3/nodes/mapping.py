from V3.env import GraphState
from V3.tools import icd11_tool
from typing import Dict, Any
from helpers.classes import ChatMessage

def mapping_node(state: GraphState) -> Dict[str, Any]:
    user_message = state.messages[-1].content
    result = icd11_tool._run({"context": state.context, "concept": user_message})
    return {
        "messages": state.messages + [ChatMessage(type="ai", content=f"[ICD11 Mapping]\n{result}")]
    }