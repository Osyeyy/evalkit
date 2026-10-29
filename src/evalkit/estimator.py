
dtype_bits = {
    "fp32": 32, 
    "fp16": 16,
    "bf16": 16,
    "int8": 8,
    "int4": 4
}

def check_compatibility(model_size_gib, free_vram_gib):
    return model_size_gib <= free_vram_gib


def estimate_weights_memory(num_parameters, dtype):
    bits = dtype_bits.get(dtype)

    if bits is None:
        raise ValueError(f"Unsupported dtype: {dtype}")

    memory_gib = (num_parameters * bits) / (8 * 1024 ** 3)

    return memory_gib

