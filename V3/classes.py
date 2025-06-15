from pydantic import BaseModel
from typing import Literal, List, Any


class ChatMessage(BaseModel):
    type: Literal["human", "tool", "ai"]
    content: str


class NamedConcept(BaseModel):
    name: str
    text: str


class NamedMemory(BaseModel):
    name: str
    # Memory items may store arbitrary data (e.g. lists of search hits)
    # so allow any type rather than restricting to dict
    content: Any


class GraphState(BaseModel):
    messages: List[ChatMessage] = []
    clinical_concept_input: str = ""
    context: List[NamedConcept] = []
    task_memory: List[NamedMemory] = []
    partial_output_code: str = ""
    final_code: str = ""


class GraphStateManager:
    def __init__(self, state: GraphState = GraphState(messages=[])):
        self.state: GraphState = state

    def update(self, data: dict) -> GraphState:
        # Atualiza apenas as chaves que tem no dict, sepre tratando  mensangens, contexto e memória como incremental e nunca substituindo
        if "messages" in data:
            data["messages"] = self.state.messages + [
                message if isinstance(message, ChatMessage) else ChatMessage(**message)
                for message in data["messages"]
            ]
        if "context" in data:
            data["context"] = self.state.context + [
                concept if isinstance(concept, NamedConcept) else NamedConcept(**concept)
                for concept in data["context"]
            ]
        if "task_memory" in data:
            data["task_memory"] = self.state.task_memory + [
                memory if isinstance(memory, NamedMemory) else NamedMemory(**memory)
                for memory in data["task_memory"]
            ]
        self.state = self.state.copy(update=data)
        print("📨 Updated state:", self.state)
        return self.state
