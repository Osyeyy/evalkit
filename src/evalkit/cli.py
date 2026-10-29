from evalkit.discovery import discover_model
from evalkit.inspector import read_json, extract_model_info, inspect_weights_index
from evalkit.estimator import check_compatibility
from evalkit.hardware import hardware_specs


def main():
    model_id = input("Which model do you want to check? ").strip()
    files_info = discover_model(model_id)


    config_path = files_info["config.json"]
    index_path = files_info["model.safetensors.index.json"]

    for filename, path in files_info.items():
        if path is not None:
            print(f"OK {filename}")
        else:
            print(f"Missing {filename}")


    if config_path is not None:
        config_output = read_json(config_path)
        model_info = extract_model_info(config_output)
        print(model_info)
    else:
        print("config.json not found")

    if index_path is not None:
        index_output = read_json(index_path)
        weights_index = inspect_weights_index(index_output)
        print(weights_index)
    else:
        print("Weights index not found")


    hardware_info = hardware_specs()

    if index_path is not None:
        model_size_gib = weights_index["total_size_gib"]
        free_vram_gib = hardware_info["free_vram_gib"]

        if model_size_gib is not None and free_vram_gib is not None:
            compatible = check_compatibility(model_size_gib, free_vram_gib)
            print("weights fit in available vram: ", compatible)
        else:
            print("cannot estimate compatibility: missing memory data")


if __name__ == "__main__":
    main()