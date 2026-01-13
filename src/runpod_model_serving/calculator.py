import math

def compute_model_vram_gb(total_params_b, quant):
    """
    model_weight_size = num_of_model_parameter * size_per_param
    Typically, size_per_param is determined by different model_weight_quantization method
    """
    # size_per_param = raw_value + index_stuff (3/AWQ-group)
    # Based on formulas.ts and llm_calc.md
    if quant == 'fp16':
        bytes_per_param = 2
        quant_factor = 1
    elif quant == 'fp8':
        bytes_per_param = 1
        # Safety factor for fp8 / int4 worst-case: ((0.5 + 3 / 32) / 0.5)
        quant_factor = (0.5 + 3 / 32) / 0.5
    elif quant == 'int8':
        bytes_per_param = 1
        quant_factor = 1
    elif quant == 'int4':
        bytes_per_param = 0.5
        # Safety factor for fp8 / int4 worst-case: ((0.5 + 3 / 32) / 0.5)
        quant_factor = (0.5 + 3 / 32) / 0.5
    else:
        bytes_per_param = 2
        quant_factor = 1

    return total_params_b * bytes_per_param * quant_factor


def compute_kv_cache_vram_gb(layers, num_kv_heads, head_dim, max_length, quant):
    """
    kvcache_size_per_req = head_dim * num_kv_head * layer * 2 * num_max_length / 2^30 * quantization_ratio
    """
    bytes_per_value = {
        'fp16': 2,
        'fp8': 1,
        'int8': 1,
        'int4': 0.5
    }.get(quant, 2)

    # Formula from llm_calc.md: head_dim * num_kv_head * layer * 2 * num_max_length / 2^30 * quantization_ratio
    # Note: 2^30 is 1024^3 (GB)
    total_bytes = head_dim * num_kv_heads * layers * 2 * max_length * bytes_per_value
    return total_bytes / (1024**3)

def calculate_performance(gpu, model_params, quant, kv_quant, max_length, user_count, parallel_gpus=1, vram_util=0.9, min_reserve_gb=2):
    model_vram = compute_model_vram_gb(model_params['total_params_b'], quant)
    kv_cache_vram_one = compute_kv_cache_vram_gb(
        model_params['layers'], 
        model_params['num_kv_heads'], 
        model_params['head_dim'], 
        max_length, 
        kv_quant
    )
    
    total_gpu_vram = gpu['vramGb'] * parallel_gpus
    
    # vLLM logic:
    # 1. Managed Pool = Total VRAM * gpu_memory_utilization
    managed_pool = total_gpu_vram * vram_util
    
    # 2. Activation Overhead Estimation
    # vLLM profiles this. For a rough estimate:
    # It depends on hidden_size, max_num_seqs, etc.
    # We'll use a heuristic: ~5-10% of model weights or min 1GB, max 4GB per GPU
    activation_overhead = max(1.0, min(4.0, model_vram * 0.1)) * parallel_gpus
    
    # 3. Available for KV Cache
    usable_kv_vram = max(0, managed_pool - model_vram - activation_overhead)
    
    # 4. Concurrency (how many full contexts fit)
    vllm_max_concurrency = usable_kv_vram / kv_cache_vram_one if kv_cache_vram_one > 0 else 0
    
    # System Reserved (outside vLLM managed pool)
    system_reserved = total_gpu_vram - managed_pool
    
    total_vram_req = model_vram + kv_cache_vram_one + activation_overhead
    
    if total_vram_req > managed_pool:
        return {
            "error": f"Insufficient VRAM: Need {total_vram_req:.2f}GB (incl. overhead) but managed pool is {managed_pool:.2f}GB",
            "success": False,
            "total_vram_req": total_vram_req,
            "usable_vram": managed_pool,
            "model_vram": model_vram,
            "kv_cache_vram": kv_cache_vram_one,
            "activation_overhead": activation_overhead,
            "reserved_vram": system_reserved
        }
        
    # Scaling factors for multi-GPU
    pp_scaling = math.pow(parallel_gpus, 0.6)
    membw_scaling = math.pow(parallel_gpus, 0.8)
    
    process_power_fp16 = gpu['processPower'].get('fp16', 0) * pp_scaling
    memory_bandwidth = gpu['memoryBandwidthGBs'] * membw_scaling
    
    active_params = model_params['active_params_b']
    total_params = model_params['total_params_b']
    
    quant_bytes = {'fp16': 2, 'fp8': 1, 'int8': 1, 'int4': 0.5}.get(quant, 2)
    
    # prompt_speed = gpu_pp / full_parameters * 1000 / sqrt(2)
    prompt_speed = (process_power_fp16 * 1000) / (total_params * math.sqrt(2))
    
    # generate_speed = gpu_membw / active_parameters / quantization_ratio
    gen_speed = memory_bandwidth / (active_params * quant_bytes)
    
    shared_prompt = prompt_speed / user_count
    shared_gen = gen_speed / user_count
    
    max_tokens = max_length * vllm_max_concurrency
    
    return {
        "success": True,
        "total_vram_req": total_vram_req,
        "model_vram": model_vram,
        "kv_cache_vram": kv_cache_vram_one,
        "activation_overhead": activation_overhead,
        "gen_speed": gen_speed,
        "prompt_speed": prompt_speed,
        "shared_gen": shared_gen,
        "shared_prompt": shared_prompt,
        "max_tokens": max_tokens,
        "full_length_gen_count": vllm_max_concurrency,
        "usable_vram": managed_pool,
        "reserved_vram": system_reserved,
        "error": None
    }
