#!/bin/bash
set -e

echo "✅ Clonando llama.cpp..."
git clone https://github.com/ggerganov/llama.cpp

echo "📥 Convertendo modelo fundido para formato GGUF..."
python llama.cpp/convert_hf_to_gguf.py ./merged-cid11-8b --outfile ggml-cid11-8b-f32.gguf

cd llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ../..

echo "⚙️ Quantizando para 4-bit k-quat (q4_K)..."
./llama.cpp/build/bin/llama-quantize ggml-cid11-8b-f32.gguf ggml-cid11-8b-q4_k.gguf q4_K

echo "✅ Conversão e quantização finalizadas com sucesso!"
