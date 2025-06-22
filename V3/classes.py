from pydantic import BaseModel
from typing import Literal, List, Any


class ChatMessage(BaseModel):
    type: Literal["human", "tool", "ai"]
    content: str


class NamedConcept(BaseModel):
    name: str
    content: Any


class GraphState(BaseModel):
    messages: List[ChatMessage] = []
    clinical_concept_input: str = ""
    context: List[NamedConcept] = []
    task_memory: List[NamedConcept] = []
    partial_output_code: str = ""
    final_code: str = ""  # Código final desse passo
    final_codes: List[str] = []  # Armazenamento de códigos finais


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
                concept
                if isinstance(concept, NamedConcept)
                else NamedConcept(**concept)
                for concept in data["context"]
            ]
        if "task_memory" in data:
            data["task_memory"] = self.state.task_memory + [
                memory if isinstance(memory, NamedConcept) else NamedConcept(**memory)
                for memory in data["task_memory"]
            ]
        if "final_codes" in data:
            data["final_codes"] = self.state.final_codes + [data["final_codes"]]
        self.state = self.state.copy(update=data)
        return self.state

    def clear_steps(self) -> GraphState:
        task_memory = [t for t in self.state.task_memory if t.name != "step"]
        state_dict = self.state.dict()
        state_dict["task_memory"] = task_memory
        return GraphState(**state_dict)
    
    def reset(self) -> GraphState:
        return GraphState()
