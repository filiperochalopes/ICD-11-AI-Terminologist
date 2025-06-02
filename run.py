from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from model_loader import load_model
from typing import TypedDict
from llama_cpp import Llama

# Define o estado compartilhado
AgentState = dict

# Carrega o modelo GGUF otimizado para CPU
llm: Llama = load_model()
print("🧠 Model loaded:", llm)

def generate_response(state: AgentState) -> tuple[str, AgentState]:
    last_input = state["input"]
    prompt = f"<s>[INST] Map the clinical concept to its ICD-11 code: {last_input} [/INST]"
    output = llm(prompt, max_tokens=512, temperature=0.2)
    result_text = output["choices"][0]["text"].strip()
    print("🤖 Response:", result_text)
    return END, {"input": last_input, "output": result_text}

# Define o grafo
builder = StateGraph(AgentState)
builder.add_node("llama_reply", generate_response)
builder.set_entry_point("llama_reply")
builder.add_edge("llama_reply", END)

# Executa
app = builder.compile()

if __name__ == "__main__":
    print("📥 Type a concept to map to ICD-11 (type 'exit' to quit)")
    while True:
        user_input = input("🧑 Concept: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        inputs = {"input": user_input, "output": ""}
        result = app.invoke(inputs)[1]  # retorna (END, state)
        print("✅ Mapped:", result["output"])
        print("---")
