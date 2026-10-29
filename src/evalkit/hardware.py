#fp32, fp16/bf16, int8, int4
import torch

def hardware_specs():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)

        return {
            "device_name": device.name,
            "device_type": "cuda",
            "total_vram_gib": total_bytes / (1024 ** 3),
            "free_vram_gib": free_bytes / (1024 ** 3)
        }

    return {
        "device_name": None,
        "device_type": "cpu",
        "total_vram_gib": None,
        "free_vram_gib": None
    }

