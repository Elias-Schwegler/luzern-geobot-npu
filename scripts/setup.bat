@echo off
REM Setup x64 Python environment for ONNX model conversion on ARM64 Windows
REM ONNX quantization tools require x86_64 Python (runs via emulation on ARM64)

echo === Qwen 3.5 NPU Setup ===
echo.

REM Check if x64 Python exists
if exist "C:\Python312-x64\python.exe" (
    echo [OK] x64 Python found at C:\Python312-x64\python.exe
) else (
    echo [!] x64 Python NOT found at C:\Python312-x64\python.exe
    echo.
    echo Please install Python 3.12 x64 from:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Choose "Windows installer (64-bit)" NOT "ARM64"
    echo Install to: C:\Python312-x64
    echo.
    pause
    exit /b 1
)

REM Create x64 virtual environment
echo Creating x64 virtual environment...
if not exist ".venv-x64" (
    C:\Python312-x64\python.exe -m venv .venv-x64
    echo [OK] Created .venv-x64
) else (
    echo [OK] .venv-x64 already exists
)

REM Activate and install dependencies
echo Installing dependencies (this may take a while)...
call .venv-x64\Scripts\activate.bat

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece protobuf
pip install onnxruntime-genai
pip install olive-ai

echo.
echo === Setup Complete ===
echo.
echo Next steps:
echo   1. Install Qualcomm QNN SDK from https://aihub.qualcomm.com
echo   2. Run: scripts\convert.bat
