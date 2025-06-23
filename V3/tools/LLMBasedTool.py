import re
from langchain.tools import BaseTool
from pydantic import PrivateAttr
from typing import ClassVar, List, Tuple, Union, Any, Dict
from langchain_community.chat_models import ChatLlamaCpp
from langchain_openai import ChatOpenAI
from V3.env import GGUF_MODEL_PATH, OPENROUTER_API_KEY, OPENROUTER_MODEL

class ReasoningResponse:
    def __init__(self, content: str, reasoning: str):
        self.content = content
        self.reasoning = reasoning

class LLMBasedTool(BaseTool):
    name: ClassVar[str] = "llm_based_tool"
    description: ClassVar[str] = (
        "Tool que usa OpenRouter (API) ou LlamaCpp local para comparar especificidade de conceitos médicos"
    )

    _llm: Union[ChatLlamaCpp, ChatOpenAI] = PrivateAttr(default=None)

    @property
    def llm(self) -> Union[ChatLlamaCpp, ChatOpenAI]:
        if self._llm is not None:
            return self._llm

        print("📥 Loading LLM model...")
        if OPENROUTER_API_KEY:
            print("📥 Loading OpenRouter LLM model with reasoning tokens...")
            self._llm = ChatOpenAI(
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1",
                model_name=OPENROUTER_MODEL,
                temperature=0.1,
                max_tokens=2560
            )
        else:
            print("📥 Loading local LlamaCpp model...")
            self._llm = ChatLlamaCpp(
                model_path=GGUF_MODEL_PATH,
                max_tokens=8 * 512,
                temperature=0.1,
                n_ctx=1536,
                verbose=True,
            )
        return self._llm

    def llm_invoke(self, prompt: str, **kwargs) -> ReasoningResponse:
        # Executa LLM e captura saída bruta
        res = self.llm.invoke(prompt, **kwargs)
        content = ""
        reasoning = ""
        # 1) Tenta extrair diretamente do llm_output conforme OpenRouter JSON
        llm_output: Dict[str, Any] = getattr(res, 'llm_output', {}) or {}
        choices = llm_output.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            reasoning = message.get('reasoning', '')
            content = message.get('content', '')
        # 2) Fallback: se não vier do JSON e content vazio, extrai de <think>
        if not content:
            full_text = res.content or ""
            tags = re.findall(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
            reasoning = "\n".join(tag.strip() for tag in tags).strip()
            content = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
            # se não sobrou conteúdo, usa reasoning como content
            if not content and reasoning:
                content, reasoning = reasoning, ""
        return ReasoningResponse(content=content.strip(), reasoning=reasoning.strip())

    def convert_llm_response_to_langgraph_messages(
        self, res: ReasoningResponse, title: str = "AI"
    ) -> List[dict]:
        if res.reasoning:
            return [
                {"type": "ai", "content": f"[Reasoning]\n{res.reasoning}"},
                {"type": "ai", "content": f"[{title}]\n{res.content}"},
            ]
        return [{"type": "ai", "content": f"[{title}]\n{res.content}"}]

    def format_prompt(
        self,
        messages: Union[str, List[Tuple[str, str]]]
    ) -> str:
        # Se LLM é via API, retorna texto puro
        if isinstance(self._llm, ChatOpenAI) or isinstance(self.llm, ChatOpenAI):
            if isinstance(messages, list):
                return "".join(text for _, text in messages)
            return messages

        # Para LlamaCpp local, aplica formatações
        model_id = ""
        try:
            model_id = (
                getattr(self.llm, "model_path", "") or getattr(self.llm, "model_name", "")
            ).lower()
        except Exception:
            pass
        # R1-0528 local
        if "r1-0528" in model_id:
            return "".join(f"<|{role}|>{text}" for role, text in messages)
        # Mistral / outras variantes locais
        if any(x in model_id for x in ["mistral", "llama"]):
            body = "".join(text for _, text in messages)
            return f"<s>[INST] {body}[/INST]"
        # Padrão roles puros
        return "".join(f"<|{role}|>{text}" for role, text in messages)

    def _run(self, prompt: str, **kwargs):
        return self.llm.invoke(prompt, **kwargs)

    async def _arun(self, prompt: str, **kwargs):
        return self._run(prompt, **kwargs)