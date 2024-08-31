import logging
try:
    import nvgpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def is_gpu_available():
    if not GPU_AVAILABLE:
        logging.warning("GPU support not available: nvgpu module not found")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": "nvgpu module not found"}
    try:
        gpu_info = nvgpu.gpu_info()
        num_gpus = len(gpu_info)
        if num_gpus == 0:
            logging.warning("No GPUs found on the system")
            return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0}
        first_gpu_vram = gpu_info[0]['mem_total']
        total_vram = sum(gpu['mem_total'] for gpu in gpu_info)
        logging.info(f"GPU(s) found: {num_gpus}, Total VRAM: {total_vram} MB")
        return {"gpu_found": True, "num_gpus": num_gpus, "first_gpu_vram": first_gpu_vram, "total_vram": total_vram, "gpu_info": gpu_info}
    except Exception as e:
        logging.error(f"Error checking GPU availability: {e}")
        return {"gpu_found": False, "num_gpus": 0, "first_gpu_vram": 0, "total_vram": 0, "error": str(e)}
