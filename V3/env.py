from langgraph.graph import StateGraph
from V3.classes import GraphState
from helpers.model_loader import load_qdrant_client
from sentence_transformers import SentenceTransformer
import torch

GGUF_MODEL_PATH = "models/ggml-icd11-8b-q4_k.gguf"
# GGUF_MODEL_PATH = "models/Ministral-8B-Instruct-2410-Q4_K_L.gguf"
# GGUF_MODEL_PATH = "models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
# GGUF_MODEL_PATH = "models/llama-2-7b-chat.Q4_K_M.gguf"


device = "cuda" if torch.cuda.is_available() else "cpu"


class Collection:
    def __init__(self, name: str, uri: str):
        self.name = name
        self.uri = uri
        self.model = self.generate_model()

    def generate_model(self):
        return SentenceTransformer(self.uri, device=device)

    def embed(self, text: str):
        return self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()


collections = [
    Collection("icd11_concepts_mpnet", "sentence-transformers/all-mpnet-base-v2"),
    Collection("icd11_concepts_biobert", "pritamdeka/S-BioBert-snli-multinli-stsb"),
    Collection(
        "icd11_concepts_sapbert", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    ),
]


TOP_K = 3

qdrant_client = load_qdrant_client()

builder = StateGraph(GraphState)
