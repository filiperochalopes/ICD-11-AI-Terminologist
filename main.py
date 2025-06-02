from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models import ChatLlamaCpp
from typing import Annotated
from pydantic import BaseModel, Field

# Caminho para o modelo GGUF personalizado
GGUF_MODEL_PATH = "models/ggml-icd11-8b-q4_k.gguf"  # Substitua pelo caminho real do seu arquivo GGUF

# Carrega o modelo GGUF com ChatLlamaCpp
llm = ChatLlamaCpp(
    model_path=GGUF_MODEL_PATH,
    max_tokens=100,
    temperature=0.2,
    n_ctx=2048,  # Ajuste conforme necessário
    verbose=True
)
print("🧠 Model loaded:", llm)

# Ferramenta de mapeamento CID-11
class ICD11ToolInput(BaseModel):
    concept: Annotated[str, Field(description="Clinical concept to map to ICD-11")]

def icd11_tool_executor(input: ICD11ToolInput) -> str:
    """
    Map a clinical concept to its ICD-11 code.
    Adds any necessary extensions or cluster codes.
    """
    prompt = f"<s>[INST] Map the clinical concept to its ICD-11 code. Add extensions or cluster codes if needed: {input.concept} [/INST]"
    output = llm.invoke(prompt)
    return output.content.strip()

# Define agente com create_react_agent
agent = create_react_agent(
    model=llm,
    tools=[icd11_tool_executor],
    prompt="You are a medical assistant specialized in mapping clinical concepts to ICD-11 codes."
)

# Exporta app
app = agent