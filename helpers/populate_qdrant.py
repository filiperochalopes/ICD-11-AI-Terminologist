import os
import time
import json
import gdown
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from tqdm import tqdm
import torch

# URL to the JSON data on Google Drive
JSON_URL = "https://drive.google.com/uc?id=1W-F3bXcgQ34djSBCoSfwClDxRSTSSEml"
JSON_PATH = "icd11_vector_input.json"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
BATCH_SIZE = 64

model_infos = [
    ("sentence-transformers/all-MiniLM-L6-v2", "icd11_concepts_minilm"),
    ("sentence-transformers/all-mpnet-base-v2", "icd11_concepts_mpnet"),
    ("pritamdeka/S-BioBert-snli-multinli-stsb", "icd11_concepts_biobert"),
    (
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "icd11_concepts_sapbert",
    ),
]


def download_json():
    if os.path.exists(JSON_PATH):
        return JSON_PATH
    print(f"[↓] Downloading JSON to {JSON_PATH}...")
    gdown.download(JSON_URL, JSON_PATH, quiet=False)
    return JSON_PATH


def generate_points(data):
    points = []
    for item in data:
        points.append(
            (
                item["concept_name"],
                {"concept_name": item["concept_name"], **item["metadata"]},
            )
        )
        for option in item["metadata"].get("postcoordination_options", []):
            points.append(
                (
                    option["title"],
                    {
                        "concept_name": item["concept_name"],
                        "parent_code": item["metadata"]["code"],
                        "postcoordination_title": option["title"],
                        **option,
                    },
                )
            )
    return points


if __name__ == "__main__":
    json_path = download_json()
    with open(json_path, "r") as f:
        data = json.load(f)

    all_points = generate_points(data)

    for model_name, collection_name in model_infos:
        print(f"\n--- Starting for {model_name} -> {collection_name} ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )

        for field in ["code", "code_type", "name_type", "is_leaf"]:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema="keyword" if field != "is_leaf" else "bool",
            )

        start = time.time()
        batch_texts, batch_payload, batch_ids = [], [], []
        for idx, (text, payload) in enumerate(
            tqdm(all_points, desc=f"Vectorizing [{collection_name}]")
        ):
            batch_texts.append(text)
            batch_payload.append(payload)
            batch_ids.append(idx)

            if len(batch_texts) >= BATCH_SIZE or idx == len(all_points) - 1:
                vectors = model.encode(
                    batch_texts, batch_size=BATCH_SIZE, show_progress_bar=False
                )
                points = [
                    PointStruct(id=batch_ids[i], vector=vectors[i].tolist(), payload=batch_payload[i])
                    for i in range(len(batch_texts))
                ]
                client.upsert(collection_name=collection_name, points=points)
                batch_texts, batch_payload, batch_ids = [], [], []

        duration = time.time() - start
        print(f"✅ [{collection_name}] Inserted {len(all_points)} points in {duration:.2f} seconds.")

    print("🏁 Completed processing for all collections.")
