@echo off
REM Convert Qwen 3.5 4B to ONNX with int4 quantization for QNN NPU
REM Must run in x64 Python environment (.venv-x64)

echo === Converting Qwen 3.5 4B to ONNX+QNN ===
echo.

call .venv-x64\Scripts\activate.bat

REM Step 1: Try ORT-GenAI model builder (preferred path)
echo [Step 1] Attempting ORT-GenAI Model Builder...
echo Command: python -m onnxruntime_genai.models.builder -m Qwen/Qwen3.5-4B-Instruct -o models/qwen35-4b-npu -p int4 -e qnn
echo.

python -m onnxruntime_genai.models.builder ^
  -m Qwen/Qwen3.5-4B-Instruct ^
  -o models/qwen35-4b-npu ^
  -p int4 ^
  -e qnn

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Model Builder failed. Trying Olive fallback...
    echo.

    REM Step 2: Fallback — use Olive auto-opt
    echo [Step 2] Attempting Olive auto-opt...
    olive auto-opt ^
      --model_name_or_path Qwen/Qwen3.5-4B-Instruct ^
      --device npu ^
      --provider QNNExecutionProvider ^
      --use_model_builder ^
      --precision int4 ^
      --output_path models/qwen35-4b-npu ^
      --log_level 1

    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [!] Olive also failed. Trying CPU-only ONNX export...
        echo.

        REM Step 3: Last resort — CPU ONNX (no NPU accel, but at least it works)
        echo [Step 3] CPU-only ONNX export (no NPU acceleration)...
        python -m onnxruntime_genai.models.builder ^
          -m Qwen/Qwen3.5-4B-Instruct ^
          -o models/qwen35-4b-cpu ^
          -p int4 ^
          -e cpu

        if %ERRORLEVEL% NEQ 0 (
            echo.
            echo [FAILED] All conversion methods failed.
            echo Check: https://onnxruntime.ai/docs/genai/howto/build-models-for-snapdragon.html
            pause
            exit /b 1
        ) else (
            echo [OK] CPU ONNX export succeeded (no NPU acceleration)
            echo Output: models/qwen35-4b-cpu
        )
    ) else (
        echo [OK] Olive conversion succeeded!
        echo Output: models/qwen35-4b-npu
    )
) else (
    echo [OK] Model Builder conversion succeeded!
    echo Output: models/qwen35-4b-npu
)

echo.
echo === Conversion Complete ===
echo.
echo Next: python server/app.py --model models/qwen35-4b-npu --port 5000
