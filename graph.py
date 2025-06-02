from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.runnables import Runnable
from typing import TypedDict, List, Annotated

from model_loader import load_model
from prompt_tool import format_prompt

# Carrega modelo local quantizado
llm = load_model()

# Define o estado da máquina
class AgentState(TypedDict):
    messages: List[str]  # histórico de mensagens
    input: str

# Função principal de inferência
async def call_model(state: AgentState) -> AgentState:
    prompt = format_prompt(state["input"])
    response = llm(prompt, max_tokens=256)
    answer = response["choices"][0]["text"].strip()

    return {
        "input": state["input"],
        "messages": state["messages"] + [answer],
    }

# Constroi grafo
workflow = StateGraph(AgentState)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")
workflow.set_finish_point("model")

graph = workflow.compile()
