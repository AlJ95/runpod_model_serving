import runpod
import os
import re
from .utils.gpu_data import GPU_CARDS
from .calculator import calculate_performance

class RunpodManager:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if self.api_key:
            runpod.api_key = self.api_key
            
    def find_best_gpu(self, model_params, quant='int4', kv_quant='fp8', max_length=8192, user_count=1, gpu_filter=None):
        best_setup = None
        min_total_price = float('inf')
        
        # Try different GPU counts (1 to 8)
        for gpu_count in range(1, 9):
            for gpu in GPU_CARDS:
                # Apply GPU filter if provided
                if gpu_filter:
                    if not re.search(gpu_filter, gpu['name'], re.IGNORECASE):
                        continue

                res = calculate_performance(
                    gpu, model_params, quant, kv_quant, max_length, user_count, parallel_gpus=gpu_count
                )
                
                if res['success'] and res['full_length_gen_count'] >= user_count:
                    # Calculate total price per hour for this setup
                    total_price = gpu.get('price_hr', 999.0) * gpu_count
                    
                    # We want the setup with the lowest total price
                    if total_price < min_total_price:
                        min_total_price = total_price
                        best_setup = {
                            "gpu": gpu,
                            "count": gpu_count,
                            "details": res,
                            "total_price": total_price
                        }
                    # If price is same, prefer fewer GPUs
                    elif abs(total_price - min_total_price) < 0.001:
                        if best_setup and gpu_count < best_setup['count']:
                            best_setup = {
                                "gpu": gpu,
                                "count": gpu_count,
                                "details": res,
                                "total_price": total_price
                            }
                            
        return best_setup

    def deploy_pod(self, gpu_name, model_id, template_id=None, gpu_count=1, pod_name="llm-serving-pod", max_model_len=8192, gpu_util=0.9, model_size_gb=20, extra_vllm_args=None):
        """
        Deploys a pod using an existing template or the official vLLM image.
        """
        # Ensure pod_name is lowercase and not None
        pod_name = (pod_name or "llm-serving-pod").lower()

        # Find the GPU in GPU_CARDS to get the correct runpod_id
        target_gpu_id = None
        for card in GPU_CARDS:
            if card['name'] == gpu_name:
                target_gpu_id = card.get('runpod_id')
                break
        
        if not target_gpu_id:
            print(f"Warning: GPU '{gpu_name}' not found in GPU_CARDS. Falling back to name.")
            target_gpu_id = gpu_name
            
        try:
            if template_id:
                pod = runpod.create_pod(
                    name=pod_name,
                    gpu_type_id=target_gpu_id,
                    template_id=template_id,
                    gpu_count=gpu_count
                )
            else:
                print(f"Deploying pod with {gpu_count}x {target_gpu_id} using vLLM image...")
                # Calculate disk size: model size + 30GB buffer for OS/vLLM
                disk_size = int(model_size_gb * 1.25 + 30)
                # Use official vLLM image
                vllm_cmd = f"--model {model_id} --gpu-memory-utilization {gpu_util} --max-model-len {max_model_len} -tp {gpu_count}"
                if extra_vllm_args:
                    vllm_cmd += f" {extra_vllm_args}"
                
                pod = runpod.create_pod(
                    name=pod_name,
                    image_name="vllm/vllm-openai:latest",
                    gpu_type_id=target_gpu_id,
                    gpu_count=gpu_count,
                    docker_args=vllm_cmd,
                    ports="8000/http",
                    container_disk_in_gb=disk_size
                )
            return pod
        except Exception as e:
            print(f"Error creating pod: {e}")
            return None

    def get_connection_details(self, pod_id):
        try:
            pod = runpod.get_pod(pod_id)
            if not pod:
                return None
            
            details = {
                "id": pod['id'],
                "name": pod.get('name'),
                "status": "UNKNOWN",
                "url": f"https://{pod['id']}-8000.proxy.runpod.net/"
            }

            if pod.get('runtime'):
                details["status"] = pod['runtime'].get('status')
            
            return details
        except Exception as e:
            print(f"Error getting pod details: {e}")
            return None

    def terminate_pod(self, pod_id):
        try:
            runpod.terminate_pod(pod_id)
            return True
        except Exception as e:
            print(f"Error terminating pod: {e}")
            return False
