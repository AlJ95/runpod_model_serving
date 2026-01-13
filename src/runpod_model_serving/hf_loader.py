from huggingface_hub import HfApi
import json
import os

def get_model_params(model_id):
    api = HfApi()
    try:
        # Try to get config.json
        config_path = api.hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Handle complex architectures like Qwen3 Omni Moe
        # It has nested configs: talker_config, thinker_config, etc.
        # We'll try to find the main text config
        text_config = config
        if "thinker_config" in config and "text_config" in config["thinker_config"]:
            text_config = config["thinker_config"]["text_config"]
        elif "text_config" in config:
            text_config = config["text_config"]
            
        # Extract parameters
        layers = text_config.get("num_hidden_layers") or text_config.get("n_layer")
        num_kv_heads = text_config.get("num_key_value_heads") or text_config.get("num_attention_heads") or text_config.get("n_head")
        
        # head_dim is often explicit now, or calculated from hidden_size / num_attention_heads
        head_dim = text_config.get("head_dim")
        if not head_dim:
            hidden_size = text_config.get("hidden_size", 4096)
            attn_heads = text_config.get("num_attention_heads") or 1
            head_dim = hidden_size // attn_heads
        
        # Estimate total params
        total_params_b = config.get("num_parameters", 0) / 1e9
        if total_params_b == 0:
            # Fallback: try to get from model info
            try:
                model_info = api.model_info(model_id)
                if hasattr(model_info, 'safetensors') and model_info.safetensors:
                    total_params_b = model_info.safetensors.get("total", 0) / 1e9
            except:
                pass
        
        # If still 0, guess from name
        if total_params_b == 0:
            if "30b" in model_id.lower(): total_params_b = 30.0
            elif "7b" in model_id.lower(): total_params_b = 7.0
            elif "8b" in model_id.lower(): total_params_b = 8.0
            elif "14b" in model_id.lower(): total_params_b = 14.0
            elif "32b" in model_id.lower(): total_params_b = 32.0
            elif "70b" in model_id.lower(): total_params_b = 70.0
            else: total_params_b = 7.0
            
        return {
            "total_params_b": total_params_b,
            "active_params_b": text_config.get("num_experts_per_tok", 1) / text_config.get("num_experts", 1) * total_params_b if "num_experts" in text_config else total_params_b,
            "layers": layers or 32,
            "num_kv_heads": num_kv_heads or 32,
            "head_dim": head_dim or 128,
            "name": model_id
        }
    except Exception as e:
        print(f"Error loading model info: {e}")
        return None
