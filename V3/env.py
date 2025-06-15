from langgraph.graph import StateGraph
from helpers.classes import GraphState
from helpers.model_loader import load_qdrant_client
from sentence_transformers import SentenceTransformer
import torch

GGUF_MODEL_PATH = "models/ggml-icd11-8b-V2-q4_k.gguf"

COLLECTION = "icd11_concepts_mpnet"

TOP_K = 3

qdrant_client = load_qdrant_client()

device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", device="cpu"
)

builder = StateGraph(GraphState)