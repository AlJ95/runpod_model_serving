import runpod
import os
from .utils.gpu_data import GPU_CARDS
from .calculator import calculate_performance

class RunpodManager:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if self.api_key:
            runpod.api_key = self.api_key
            
    def find_best_gpu(self, model_params, quant='int4', kv_quant='fp8', max_length=8192, user_count=1):
        best_setup = None
        min_total_price = float('inf')
        
        # Try different GPU counts (1 to 8)
        for gpu_count in range(1, 9):
            for gpu in GPU_CARDS:
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

    def deploy_template(self, gpu_name, template_id, gpu_count=1, pod_name="llm-serving-pod"):
        """
        Deploys a pod using an existing template.
        """
        # Mapping common names to Runpod IDs (simplified)
        gpu_id_map = {
            "NVIDIA RTX 3090": "NVIDIA GeForce RTX 3090",
            "NVIDIA RTX 4090": "NVIDIA GeForce RTX 4090",
            "NVIDIA A100 80G": "NVIDIA A100 80GB PCIe",
            "NVIDIA H100 80G": "NVIDIA H100 80GB PCIe",
        }
        
        # Clean up name for matching
        clean_name = gpu_name.replace("NVIDIA ", "").split(" ")[0]
        target_gpu_id = None
        for k, v in gpu_id_map.items():
            if clean_name in k:
                target_gpu_id = v
                break
        
        if not target_gpu_id:
            target_gpu_id = gpu_name # Fallback
            
        try:
            pod = runpod.create_pod(
                name=pod_name,
                image_name=None, # Template handles this
                gpu_type_id=target_gpu_id,
                template_id=template_id,
                gpu_count=gpu_count
            )
            return pod
        except Exception as e:
            print(f"Error creating pod: {e}")
            return None

    def get_connection_details(self, pod_id):
        try:
            pod = runpod.get_pod(pod_id)
            if pod and pod.get('runtime'):
                return {
                    "id": pod['id'],
                    "status": pod['runtime'].get('status'),
                    "ip": pod['runtime'].get('address'),
                    "ports": pod['runtime'].get('ports')
                }
            return pod
        except Exception as e:
            print(f"Error getting pod details: {e}")
            return None
