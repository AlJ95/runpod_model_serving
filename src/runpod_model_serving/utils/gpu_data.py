GPU_CARDS = [
    # Featured GPUs
    {"name": "NVIDIA RTX 5090", "runpod_id": "NVIDIA GeForce RTX 5090", "vramGb": 32, "memoryBandwidthGBs": 1790.0, "processPower": {"fp16": 104.8}, "kvQuantType": "fp8", "price_hr": 0.76},
    {"name": "NVIDIA A40", "runpod_id": "NVIDIA A40", "vramGb": 48, "memoryBandwidthGBs": 695.8, "processPower": {"fp16": 37.4}, "kvQuantType": "fp8", "price_hr": 0.20},
    {"name": "NVIDIA H200 SXM", "runpod_id": "NVIDIA H200", "vramGb": 141, "memoryBandwidthGBs": 4800.0, "processPower": {"fp16": 989.0}, "kvQuantType": "fp8", "price_hr": 3.05},
    {"name": "NVIDIA B200", "runpod_id": "NVIDIA B200", "vramGb": 180, "memoryBandwidthGBs": 8000.0, "processPower": {"fp16": 2250.0}, "kvQuantType": "fp8", "price_hr": 4.41},
    
    # NVIDIA latest generation
    {"name": "NVIDIA RTX 2000 Ada", "runpod_id": "NVIDIA RTX 2000 Ada Generation", "vramGb": 16, "memoryBandwidthGBs": 224.0, "processPower": {"fp16": 19.2}, "kvQuantType": "fp8", "price_hr": 0.18},
    {"name": "NVIDIA RTX 4000 Ada", "runpod_id": "NVIDIA RTX 4000 Ada Generation", "vramGb": 20, "memoryBandwidthGBs": 360.0, "processPower": {"fp16": 26.7}, "kvQuantType": "fp8", "price_hr": 0.20},
    {"name": "NVIDIA RTX 4090", "runpod_id": "NVIDIA GeForce RTX 4090", "vramGb": 24, "memoryBandwidthGBs": 1008.0, "processPower": {"fp16": 82.6}, "kvQuantType": "fp8", "price_hr": 0.50},
    {"name": "NVIDIA L4", "runpod_id": "NVIDIA L4", "vramGb": 24, "memoryBandwidthGBs": 300.0, "processPower": {"fp16": 30.3}, "kvQuantType": "fp8", "price_hr": 0.32},
    {"name": "NVIDIA L40S", "runpod_id": "NVIDIA L40S", "vramGb": 48, "memoryBandwidthGBs": 864.0, "processPower": {"fp16": 91.5}, "kvQuantType": "fp8", "price_hr": 0.71},
    {"name": "NVIDIA RTX 6000 Ada", "runpod_id": "NVIDIA RTX 6000 Ada Generation", "vramGb": 48, "memoryBandwidthGBs": 960.0, "processPower": {"fp16": 91.1}, "kvQuantType": "fp8", "price_hr": 0.63},
    {"name": "NVIDIA H100 SXM", "runpod_id": "NVIDIA H100 80GB HBM3", "vramGb": 80, "memoryBandwidthGBs": 3350.0, "processPower": {"fp16": 989.0}, "kvQuantType": "fp8", "price_hr": 2.69},
    {"name": "NVIDIA H100 PCIe", "runpod_id": "NVIDIA H100 PCIe", "vramGb": 80, "memoryBandwidthGBs": 2000.0, "processPower": {"fp16": 756.0}, "kvQuantType": "fp8", "price_hr": 2.03},
    {"name": "NVIDIA H100 NVL", "runpod_id": "NVIDIA H100 NVL", "vramGb": 94, "memoryBandwidthGBs": 3900.0, "processPower": {"fp16": 835.0}, "kvQuantType": "fp8", "price_hr": 2.61},
    {"name": "NVIDIA RTX PRO 6000", "runpod_id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "vramGb": 96, "memoryBandwidthGBs": 960.0, "processPower": {"fp16": 91.1}, "kvQuantType": "fp8", "price_hr": 1.56},
    {"name": "NVIDIA RTX PRO 6000 WK", "runpod_id": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "vramGb": 96, "memoryBandwidthGBs": 960.0, "processPower": {"fp16": 91.1}, "kvQuantType": "fp8", "price_hr": 1.78},

    # NVIDIA previous generation
    {"name": "NVIDIA RTX A4000", "runpod_id": "NVIDIA RTX A4000", "vramGb": 16, "memoryBandwidthGBs": 448.0, "processPower": {"fp16": 19.2}, "kvQuantType": "fp8", "price_hr": 0.19},
    {"name": "NVIDIA RTX A4500", "runpod_id": "NVIDIA RTX A4500", "vramGb": 20, "memoryBandwidthGBs": 640.0, "processPower": {"fp16": 23.7}, "kvQuantType": "fp8", "price_hr": 0.19},
    {"name": "NVIDIA RTX 3090", "runpod_id": "NVIDIA GeForce RTX 3090", "vramGb": 24, "memoryBandwidthGBs": 936.2, "processPower": {"fp16": 35.6}, "kvQuantType": "fp8", "price_hr": 0.34},
    {"name": "NVIDIA RTX A5000", "runpod_id": "NVIDIA RTX A5000", "vramGb": 24, "memoryBandwidthGBs": 768.0, "processPower": {"fp16": 27.8}, "kvQuantType": "fp8", "price_hr": 0.20},
    {"name": "NVIDIA RTX A6000", "runpod_id": "NVIDIA RTX A6000", "vramGb": 48, "memoryBandwidthGBs": 768.0, "processPower": {"fp16": 38.7}, "kvQuantType": "fp8", "price_hr": 0.40},
    {"name": "NVIDIA A100 PCIe", "runpod_id": "NVIDIA A100 80GB PCIe", "vramGb": 80, "memoryBandwidthGBs": 1935.0, "processPower": {"fp16": 312.0}, "kvQuantType": "fp8", "price_hr": 1.14},
    {"name": "NVIDIA A100 SXM", "runpod_id": "NVIDIA A100-SXM4-80GB", "vramGb": 80, "memoryBandwidthGBs": 2039.0, "processPower": {"fp16": 312.0}, "kvQuantType": "fp8", "price_hr": 1.22},
    
    # Legacy / Others
    {"name": "NVIDIA H20 96G", "runpod_id": "NVIDIA H20 96G", "vramGb": 96, "memoryBandwidthGBs": 4096.0, "processPower": {"fp16": 148.0}, "kvQuantType": "fp8", "price_hr": 1.50},
    {"name": "NVIDIA H800 80G", "runpod_id": "NVIDIA H800 80G", "vramGb": 80, "memoryBandwidthGBs": 2048.0, "processPower": {"fp16": 204.9}, "kvQuantType": "fp8", "price_hr": 2.00},
    {"name": "NVIDIA V100 32G", "runpod_id": "Tesla V100-SXM2-32GB", "vramGb": 32, "memoryBandwidthGBs": 897.0, "processPower": {"fp16": 28.3}, "kvQuantType": "fp16", "price_hr": 0.60},
    {"name": "AMD RX7900XTX 24G", "runpod_id": "AMD RX7900XTX 24G", "vramGb": 24, "memoryBandwidthGBs": 960.0, "processPower": {"fp16": 61.4}, "kvQuantType": "fp8", "price_hr": 0.40},
    {"name": "AMD R780M 8G*", "runpod_id": "AMD R780M 8G*", "vramGb": 8, "memoryBandwidthGBs": 89.6, "processPower": {"fp16": 16.6}, "kvQuantType": "fp8", "price_hr": 0.10},
]
