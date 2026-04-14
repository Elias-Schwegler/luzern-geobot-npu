"""
OpenAI-compatible API server for ONNX Runtime GenAI models.

Serves Qwen 3.5 4B (ONNX+QNN) on Snapdragon X NPU with /v1/chat/completions.
LiteLLM connects to this as an openai/ provider.

Usage:
  python server/app.py --model models/qwen35-4b-npu --port 5000
  python server/app.py --model models/qwen35-4b-npu --port 5000 --provider qnn
"""

import argparse
import json
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Qwen 3.5 NPU Server", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global model reference (set at startup)
_model = None
_tokenizer = None
_model_name = "qwen3.5-4b-npu"


# ---------------------------------------------------------------------------
# Request/Response models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3.5-4b-npu"
    messages: list[ChatMessage]
    temperature: float = 0.3
    max_tokens: int = 1024
    stream: bool = False


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, provider: str = "qnn"):
    """Load ONNX model with specified execution provider."""
    global _model, _tokenizer

    try:
        import onnxruntime_genai as og

        print(f"Loading model from: {model_path}")
        print(f"Execution provider: {provider}")

        if provider == "qnn":
            config = og.Config(model_path)
            config.clear_providers()
            config.append_provider("qnn")
            _model = og.Model(config)
        else:
            _model = og.Model(model_path)

        _tokenizer = og.Tokenizer(_model)
        print(f"Model loaded successfully!")
        return True

    except ImportError:
        print("ERROR: onnxruntime_genai not installed.")
        print("Install: pip install onnxruntime-genai")
        return False
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print(f"Make sure the model exists at: {model_path}")
        return False


def generate(messages: list[ChatMessage], max_tokens: int = 1024, temperature: float = 0.3) -> tuple[str, int, int]:
    """Generate a response using ONNX Runtime GenAI."""
    import onnxruntime_genai as og

    # Build chat template
    # Qwen uses ChatML format: <|im_start|>role\ncontent<|im_end|>
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    # Tokenize
    input_tokens = _tokenizer.encode(prompt)
    prompt_length = len(input_tokens)

    # Generate
    params = og.GeneratorParams(_model)
    params.set_search_options(
        max_length=prompt_length + max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    params.input_ids = input_tokens

    generator = og.Generator(_model, params)
    output_tokens = []

    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        output_tokens.append(token)

    # Decode
    response_text = _tokenizer.decode(output_tokens)

    # Strip any trailing <|im_end|>
    if "<|im_end|>" in response_text:
        response_text = response_text.split("<|im_end|>")[0]

    return response_text.strip(), prompt_length, len(output_tokens)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [{"id": _model_name, "object": "model", "owned_by": "local"}],
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy" if _model is not None else "no_model", "model": _model_name}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    if _model is None:
        return {"error": "Model not loaded"}

    start = time.time()

    try:
        response_text, prompt_tokens, completion_tokens = generate(
            request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as e:
        return {"error": str(e)}

    elapsed = time.time() - start

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=_model_name,
        choices=[
            ChatChoice(
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen 3.5 NPU Server")
    parser.add_argument("--model", required=True, help="Path to ONNX model directory")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--provider", default="qnn", choices=["qnn", "cpu"], help="Execution provider")
    args = parser.parse_args()

    _model_name = f"qwen3.5-4b-{args.provider}"

    if not load_model(args.model, args.provider):
        print("Failed to load model. Exiting.")
        exit(1)

    print(f"Starting server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
