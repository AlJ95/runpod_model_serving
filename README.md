# Runpod Model Serving & VRAM Calculator

This project provides a Python-based VRAM calculator for LLMs and a tool to deploy them on Runpod.

## Features
- **VRAM Calculation**: Accurate estimation of model and KV cache memory requirements.
- **HuggingFace Integration**: Automatically fetch model parameters (layers, heads, etc.) from HuggingFace.
- **Runpod Deployment**: Find the best GPU for your model and deploy it using Runpod templates.
- **CLI Tool**: Easy to use command-line interface.

## Installation
```bash
git clone <repo-url>
cd runpod_model_serving
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Usage
### CLI
```bash
runpod-serve --model Qwen/Qwen2.5-7B-Instruct --quant int4
```

### Python API
```python
from runpod_model_serving import get_model_params, calculate_performance, RunpodManager

# Get model info
params = get_model_params("Qwen/Qwen2.5-7B-Instruct")

# Find best GPU
manager = RunpodManager(api_key="your_api_key")
best_gpu = manager.find_best_gpu(params)
print(f"Recommended GPU: {best_gpu['name']}")
```

## Development
The calculation logic is based on the `llm_calculation_project.txt` formulas.
