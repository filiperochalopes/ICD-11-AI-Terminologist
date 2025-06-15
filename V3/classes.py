from pydantic import BaseModel
from typing import Literal, List


class ChatMessage(BaseModel):
    type: Literal["human", "ai"]
    content: str


class NamedConcept(BaseModel):
    name: str
    text: str


class NamedMemory(BaseModel):
    name: str
    content: dict


class GraphState(BaseModel):
    messages: List[ChatMessage]
    clinical_concept_input: str = ""
    context: List[NamedConcept] = []
    memory: List[NamedMemory] = []
    partial_output_code: str = ""
    final_code: str = ""
