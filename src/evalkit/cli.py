from evalkit.discovery import discover_model
from evalkit.inspector import read_json, extract_model_info, inspect_weights_index

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


if __name__ == "__main__":
    main()