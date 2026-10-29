import json
from huggingface_hub import HfApi

def read_json(path):
    with open(f'{path}') as json_file:
        data = json.load(json_file)

    return data


def extract_model_info(config_output):
    text_config = config_output.get("text_config", config_output)
    model_type = text_config.get("model_type")
    architectures = config_output.get("architectures")
    hidden_size = text_config.get("hidden_size")
    num_hidden_layers = text_config.get("num_hidden_layers")
    num_attention_heads = text_config.get("num_attention_heads")
    num_key_value_heads = text_config.get("num_key_value_heads")
    vocab_size = text_config.get("vocab_size")
    intermediate_size = text_config.get("intermediate_size")
    max_position_embeddings = text_config.get("max_position_embeddings")
    head_dim = text_config.get("head_dim")
    tie_word_embeddings = text_config.get("tie_word_embeddings")


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

def inspect_weights_index(index_output):
    metadata = index_output.get("metadata", {})
    total_size = metadata.get("total_size")
    weight_map = index_output.get("weight_map", {})
    shard_count = len(set(weight_map.values()))
    tensor_count = len(weight_map)
    unique_shards = sorted(set(weight_map.values()))
    total_size_gib = (total_size / (1024 ** 3) if total_size is not None else None)

    
    return {
        "total_size": total_size,
        "shard_count": shard_count,
        "total_size_gib": total_size_gib,
        "tensor_count": tensor_count,
        "unique_shards": unique_shards
    }

def inspect_model_parameter(model_id):
    api = HfApi()
    model_info = api.model_info(model_id)

    if hasattr(model_info, 'safetensors') and model_info.safetensors is not None:
        return {
            "num_parameters": model_info.safetensors.total
        }

    return {
        "num_parameters": None
    }





