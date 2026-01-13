import sys
import argparse
import time
import requests
from .hf_loader import get_model_params
from .runpod_manager import RunpodManager
from .calculator import calculate_performance

def main():
    parser = argparse.ArgumentParser(description="Runpod LLM Serving Tool")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace Model ID")
    parser.add_argument("--template", type=str, help="Runpod Template ID")
    parser.add_argument("--api-key", type=str, help="Runpod API Key")
    parser.add_argument("--quant", type=str, default="int4", choices=["fp16", "fp8", "int8", "int4"])
    parser.add_argument("--kv-quant", type=str, default="fp8", choices=["fp16", "fp8", "int8", "int4"])
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--users", type=int, default=1)
    parser.add_argument("--util", type=float, default=0.95, help="GPU Memory Utilization (vLLM default 0.9)")
    parser.add_argument("--dry-run", action="store_true", help="Calculate only, do not deploy")
    parser.add_argument("--pod-name", type=str, help="Custom name for the pod")
    
    args = parser.parse_args()
    
    print(f"Fetching model info for {args.model}...")
    params = get_model_params(args.model)
    if not params:
        print("Failed to fetch model info.")
        sys.exit(1)
        
    print(f"Model Params: {params['total_params_b']:.2f}B parameters, {params['layers']} layers")
    
    manager = RunpodManager(api_key=args.api_key)
    # Find best GPU based on requested user count (Strict Concurrency)
    best_setup = manager.find_best_gpu(
        params, 
        quant=args.quant, 
        kv_quant=args.kv_quant, 
        max_length=args.max_length, 
        user_count=args.users
    )
    
    if not best_setup:
        print(f"No suitable GPU setup found that can handle {args.users} concurrent users.")
        sys.exit(1)
        
    gpu = best_setup['gpu']
    count = best_setup['count']
    res = best_setup['details']
    
    print(f"\n--- Best Setup found: {count}x {gpu['name']} (${best_setup['total_price']:.2f}/hr) ---")
    
    print(f"VRAM Breakdown (per GPU pool):")
    print(f"  - Model Weights: {res['model_vram']/count:.2f} GB")
    print(f"  - Activation OH: {res['activation_overhead']/count:.2f} GB (estimated)")
    print(f"  - KV Cache (1x): {res['kv_cache_vram']:.2f} GB")
    print(f"  - System Reserv: {res['reserved_vram']/count:.2f} GB (outside vLLM)")
    print(f"  - Total Managed: {res['usable_vram']/count:.2f} GB (utilization: {args.util})")
    
    print(f"\nvLLM Concurrency Estimates:")
    print(f"  - Max Concurrency: {res['full_length_gen_count']:.2f}x (at {args.max_length} tokens)")
    print(f"  - Total Capacity:  {res['max_tokens']:.0f} tokens")
    
    print(f"\nPerformance Estimates (for {args.users} users):")
    print(f"  - Prompt Speed:  {res['prompt_speed']:.0f} tok/s")
    print(f"  - Gen Speed:     {res['gen_speed']:.0f} tok/s")
    print(f"  - Per User Gen:  {res['shared_gen']:.1f} tok/s")
    
    if args.dry_run:
        print("\nDry run enabled. Skipping deployment.")
        return

    print(f"\nDeploying to Runpod...")
    pod = manager.deploy_pod(
        gpu_name=gpu['name'], 
        model_id=args.model,
        template_id=args.template, 
        gpu_count=count,
        pod_name=args.pod_name,
        max_model_len=args.max_length,
        gpu_util=args.util,
        model_size_gb=res['model_vram']
    )
    
    if pod:
        print(f"Pod created successfully! ID: {pod['id']}")
        print("Waiting for connection details and model response...")
        
        details = None
        url = f"https://{pod['id']}-8000.proxy.runpod.net/"
        
        # Wait for pod to be ready and model to respond
        max_wait_minutes = 10
        start_time = time.time()
        model_ready = False
        
        while (time.time() - start_time) < (max_wait_minutes * 60):
            details = manager.get_connection_details(pod['id'])
            if details and details.get('status') == 'RUNNING':
                try:
                    # Try to ping the vLLM health or models endpoint
                    response = requests.get(f"{url}v1/models", timeout=5)
                    if response.status_code == 200:
                        model_ready = True
                        break
                except:
                    pass
            
            print(f"  ... still waiting (elapsed: {int(time.time() - start_time)}s)", end='\r')
            time.sleep(10)
            
        print("\n")
        if details:
            print(f"Connection Details:")
            print(f"  - ID:     {details['id']}")
            print(f"  - Status: {details['status']}")
            print(f"  - URL:    {url}")
            if model_ready:
                print("  - Model:  READY (responding to ping)")
            else:
                print("  - Model:  NOT READY (timed out waiting for response)")
        else:
            print("Could not fetch connection details.")
    else:
        print("Failed to create pod.")

if __name__ == "__main__":
    main()
