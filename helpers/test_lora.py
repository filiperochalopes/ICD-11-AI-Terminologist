from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Caminho do modelo já fundido com LoRA
MODEL_PATH = "./merged-cid11-8b"

# Carregamento
print("📥 Carregando modelo fundido...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"  # usa GPU se disponível
)

# Entrada de teste
concept = "Diverticulitis of cecum"
prompt = f"<s>[INST] Map the clinical concept to its ICD-11 code. Add extensions or cluster codes if needed: {concept} [/INST]"

# Tokenização
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Geração
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

# Decodificação
decoded = tokenizer.decode(output[0], skip_special_tokens=False)

# Extração da resposta após [/INST]
if "[/INST]" in decoded:
    response = decoded.split("[/INST]")[-1].strip()
else:
    response = decoded

print("\n📨 Prompt:", concept)
print("🤖 Resposta completa:", decoded.strip())
print("✅ Resposta gerada:", response)