# Qwen 3.5 4B on Snapdragon X Elite NPU

Experimental project: run Qwen 3.5 4B on Snapdragon X Elite NPU via ONNX Runtime
GenAI with QNN Execution Provider. Replace CPU-only Ollama (~2 t/s) with
NPU-accelerated inference (~10-20 t/s estimated).

## Architecture

```
Qwen 3.5 4B (HuggingFace)
    → ORT-GenAI Model Builder (int4 quantization + QNN export)
    → ONNX model + QNN context binaries
    → FastAPI server (/v1/chat/completions, OpenAI-compatible)
    → LiteLLM connects as openai/ provider
    → Geopard backend uses it transparently
```

## Prerequisites

- Windows 11 ARM64 (Surface Laptop with Snapdragon X Elite)
- Python 3.12 x64 (ONNX quantization requires x86_64, runs via emulation on ARM64)
- Qualcomm QNN SDK (free, requires Qualcomm ID from aihub.qualcomm.com)
- ~8GB disk space for model files

## Quick Start

```bash
# 1. Setup (x64 Python venv)
scripts/setup.bat

# 2. Convert model
scripts/convert.bat

# 3. Start server
python server/app.py --model models/qwen35-4b-npu --port 5000

# 4. Test
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-4b-npu", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Status

- [ ] x64 Python environment
- [ ] ONNX Runtime GenAI + QNN SDK installed
- [ ] Qwen 3.5 4B converted to ONNX+QNN
- [ ] OpenAI-compatible server
- [ ] LiteLLM integration
- [ ] Tool calling verified
- [ ] Benchmark vs Ollama CPU
