from typing import Optional
import re
from langchain.tools import BaseTool
from pydantic import PrivateAttr
from langchain_community.chat_models import ChatLlamaCpp
from V3.env import GGUF_MODEL_PATH, OPENROUTER_API_KEY, OPENROUTER_MODEL

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # OpenAI wrapper may not be installed


class ReasoningResponse:
    def __init__(self, content, reasoning):
        self.content = content
        self.reasoning = reasoning


class LLMBasedTool(BaseTool):
    """
    Base Tool that provides a lazily‑loaded `llm` property.
    """

    _llm: ChatLlamaCpp = PrivateAttr(default=None)

    @property
    def llm(self) -> ChatLlamaCpp:
        if self._llm is not None:
            return self._llm

        openrouter_key = OPENROUTER_API_KEY
        print("📥 Loading LLM model...")
        if openrouter_key and ChatOpenAI:
            print("📥 Loading OpenRouter LLM model...")
            model_name = OPENROUTER_MODEL
            self._llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model_name=model_name,
                temperature=0.1,
                max_tokens=64,
            )
        else:
            # Fallback to local ChatLlamaCpp
            self._llm = ChatLlamaCpp(
                model_path=GGUF_MODEL_PATH,
                max_tokens=2560,
                temperature=0.1,
                n_ctx=1536,
                verbose=True,
            )

        return self._llm

    # Patch invoke to separate reasoning from final
    def llm_invoke(self, prompt, **kwargs) -> ReasoningResponse:
        orig_invoke = self._llm.invoke
        res = orig_invoke(prompt, **kwargs)
        full = res.content
        matches = re.findall(r"<think>(.*?)</think>", full, flags=re.DOTALL)
        res.reasoning = "\n".join(matches).strip()
        res.content = re.sub(r"<think>.*?</think>", "", full, flags=re.DOTALL).strip()

        return res

    def convert_llm_response_to_langgraph_messages(
        self, res: ReasoningResponse, title: str = "AI"
    ) -> list:
        if res.reasoning:
            return [
                {"type": "ai", "content": f"[Reasoning]\n{res.reasoning}"},
                {"type": "ai", "content": f"[{title}]\n{res.content.strip()}"}
            ]
        else:
            return [{"type": "ai", "content": f"[{title}]\n{res.content.strip()}"}]

    def format_prompt(self, messages):
        model_id = ""
        try:
            model_id = getattr(self.llm, "model_path", "") or getattr(
                self.llm, "model_name", ""
            )
        except Exception:
            model_id = ""
        if "r1-0528" in model_id.lower():
            # Existing R1-0528 formatting logic here
            body = ""
            for role, text in messages:
                body += f"<|{role}|>{text}"
            return body
        elif any(x in model_id.lower() for x in ["mistral", "llama"]):
            body = "".join(text for _, text in messages)
            return f"<s>[INST] {body}[/INST]"
        else:
            # Pure role tags formatting
            body = ""
            for role, text in messages:
                body += f"<|{role}|>{text}"
            return body
