from pydantic import BaseModel
from typing import Literal, List


class ChatMessage(BaseModel):
    type: Literal["human", "ai"]
    content: str

class GraphState(BaseModel):
    messages: List[ChatMessage]
    context: str = ""
