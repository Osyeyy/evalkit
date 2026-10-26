from evalkit.discovery import discover_model
from evalkit.inspector import read_config, extract_model_info
def main():
    model_id = input("Which model do you want to check?").strip()
    files_info = discover_model(model_id)
    config_path = files_info["config.json"]

    for filename, path in files_info.items():
        if path is not None:
            print(f"OK {filename}")
        else:
            print(f"Missing {filename}")
    if config_path is not None:
        config_output = read_config(config_path)
        model_info = extract_model_info(config_output)

        print(model_info)
    else:
        print("config.json not found")

    

if __name__ == "__main__":
    main() 