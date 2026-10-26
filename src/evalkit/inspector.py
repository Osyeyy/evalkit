import json

def read_json(path):
    with open(f'{path}') as json_file:
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
    max_position_embeddings = config_output.get("max_position_embeddings")
    head_dim = config_output.get("head_dim")
    tie_word_embeddings = config_output.get("tie_word_embeddings")


    return {
        "model_type": model_type,
        "architectures": architectures,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_position_embeddings,
        "head_dim": head_dim,
        "tie_word_embeddings": tie_word_embeddings
}

def inspect_weights_index(index_path):
    metadata = index_output.get("metadata", {})
    total_size = metadata.get("total_size")
    weight_map = index_output.get("weight_map", {})

    return {
        "total_size": total_size,
        "weight_map": weight_map
    }


