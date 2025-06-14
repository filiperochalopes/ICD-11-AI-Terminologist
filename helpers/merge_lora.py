# ! pip install peft transformers accelerate bitsandbytes typing_extensions==4.6.0 sentencepiece protobuf --upgrade
# apt update && apt install -y cmake build-essential libcurl4-openssl-dev git-lfs

# merge_lora.py  (execute em máquina com GPU)
from peft import PeftModel
from transformers import AutoModelForCausalLM

from huggingface_hub import login, snapshot_download
login(token="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")

snapshot_download(
    repo_id="filipelopesmedbr/cid11-agent-mistral-8b",
    repo_type="model",
    local_dir="./outputs/cid11-agent-mistral-8b",
    ignore_patterns=["ggml-icd11-8b-q4_k.gguf"]
)

BASE = "mistralai/Ministral-8B-Instruct-2410"

ADAPTER = "./outputs/cid11-agent-mistral-8b"          # seu diretório
MERGED = "./merged-cid11-8b"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="bfloat16").cuda()
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()        # aplica LoRA nos pesos

model.save_pretrained(MERGED, safe_serialization=True)

from transformers import AutoTokenizer

# use o mesmo tokenizer do modelo base
tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)

# salve na pasta do modelo mesclado
tokenizer.save_pretrained("merged-cid11-8b")