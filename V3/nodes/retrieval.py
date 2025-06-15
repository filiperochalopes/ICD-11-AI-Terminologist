from V3.env import GraphState
from V3.tools import qdrant_tool
from typing import Dict, Any
from helpers.classes import ChatMessage

def retrieval_node(state: GraphState) -> Dict[str, Any]:
    user_message = state.messages[-1].content
    print(f"🔍 User message for retrieval: {user_message}")
    result = qdrant_tool._run(user_message)
    return {
        "context": result,
        "messages": state.messages + [ChatMessage(type="ai", content=f"[Qdrant Results]\n{result}")]
    }