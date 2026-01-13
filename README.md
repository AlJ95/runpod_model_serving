# Runpod Model Serving & VRAM Calculator

This project provides a Python-based VRAM calculator for LLMs and a tool to deploy them on Runpod.

> **Note**: This project is a bit **vibecoded** but thoroughly tested. It aims to provide the most accurate hardware recommendations based on real-world vLLM memory management logic.

## Features
- **VRAM Calculation**: Accurate estimation of model and KV cache memory requirements.
- **vLLM Concurrency Logic**: Calculates maximum concurrency based on `gpu_memory_utilization` and estimated activation overhead.
- **HuggingFace Integration**: Automatically fetch model parameters (layers, heads, etc.) from HuggingFace, including support for complex architectures like Qwen3 Omni.
- **Cost-Optimized Deployment**: Finds the cheapest GPU setup (1-8 GPUs) that satisfies your concurrency requirements.
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
runpod-serve --model Qwen/Qwen3-Omni-30B-A3B-Instruct --quant int4 --users 10
```

### Python API
```python
from runpod_model_serving import get_model_params, RunpodManager

# Get model info
params = get_model_params("Qwen/Qwen3-Omni-30B-A3B-Instruct")

# Find best GPU setup for 10 concurrent users
manager = RunpodManager(api_key="your_api_key")
best_setup = manager.find_best_gpu(params, user_count=10)
print(f"Recommended Setup: {best_setup['count']}x {best_setup['gpu']['name']}")
```

## License
MIT
