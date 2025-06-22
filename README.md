# ICD-11 LangGraph Agent

This repository contains a set of LangGraph agents that map clinical concepts to ICD‑11 codes. Each version builds on the previous one, offering new retrieval strategies and model orchestration.

## Getting Started

```sh
conda create -n langgraph python=3.11 -y
conda activate langgraph
conda install -c conda-forge compilers cmake
pip install --prefer-binary -r requirements.txt
```

## Running

Start LangGraph Studio and load the application from Version 3:

```sh
langgraph dev --host 0.0.0.0 --no-browser
langgraph run V3:app
```

### Using `tmux`

```sh
# create a session
tmux new -s lg

# inside the session
conda activate langgraph
langgraph dev --host 0.0.0.0 --no-browser

# detach without stopping
Ctrl-b d

# reattach later
tmux attach -t lg

# terminate
tmux kill-session -t lg

The tmux session lets you keep the development server running while debugging in LangSmith. Copy the application URL (baseUrl) and open:
https://smith.langchain.com/studio/thread?baseUrl=<APP_URL>&mode=chat
where `<APP_URL>` is your server address.
```

## Release Notes – Version 3 (0.3.0)

- **Multi-encoder Semantic Search** using MPNet, BioBERT and SapBERT embeddings to retrieve the top‑K leaf codes from Qdrant (default `TOP_K=3`).
- **Exact FSN Matching** that bypasses the LLM when the clinical concept exactly matches a fully specified name.
- **LLM-based Code Selection** with optional local `ChatLlamaCpp` or OpenRouter backed models.
- **Hierarchical Specificity Checking** combining token heuristics and LLM analysis to ensure that the returned code matches the concept’s specificity.
- **Iterative Final Looper** that iterates through retrieved candidates until three final codes are collected or the list is exhausted.
- **LangGraph `StateGraph` Orchestration** with typed state management and dynamic prompt formatting.
- **GPU Acceleration** when CUDA is available.

Future nodes have been scaffolded for post‑coordination workflows and extension code retrieval.

## Earlier Versions

Version 1 introduced the basic two‑step pipeline—retrieval of similar stem codes from Qdrant followed by a single LlamaCpp call for mapping. Version 2 added post‑coordination instructions and generated longer responses to explain cluster formation. Version 3 builds a more robust graph architecture with multiple encoders, exact matching, specificity checking and looping.


## Qdrant

Start a local Qdrant instance:

```sh
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

Then populate the collections:

```sh
python helpers/populate_qdrant.py
```

## Docker

Build the image for a specific version (defaults to V3):

```sh
docker build -t icd11 --build-arg VERSION=V3 .
```

Run the container:

```sh
docker run -p 8000:8000 icd11
```

The container executes `langgraph run <VERSION>:app` inside.

