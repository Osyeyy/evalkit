import json

def read_config(config_path):
    with open(f'{config_path}') as json_file:
        data = json.load(json_file)

    return data

def extract_model_info(config_output):
    model_type = config_output.get("model_type")
    architectures = config_output.get("architectures")
    hidden_size = config_output.get("hidden_size")
    num_hidden_layers = config_output.get("num_hidden_layers")
    num_attention_heads = config_output.get("num_attention_heads")
    num_key_value_heads = config_output.get("num_key_value_heads")
    vocab_size = config_output.get("vocab_size")
    intermediate_size = config_output.get("intermediate_size")

    return {
    "model_type": model_type,
    "architectures": architectures,
    "hidden_size": hidden_size,
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "num_key_value_heads": num_key_value_heads,
    "vocab_size": vocab_size,
    "intermediate_size": intermediate_size
}

