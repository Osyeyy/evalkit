### tokenization, forward pass, results 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from evalkit.metrics import perplexity_n


def evaluate_model(model_id, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = 512 
    context = 384 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
    if len(ids) < 2: 
        raise ValueError("Text must have at least 2 tokens")

    total_loss = 0.0
    total_tokens = 0
    next_token = 1 

    while next_token < len(ids):
        start = max(0, next_token - context)
        end = min(start + window, len(ids))
        chunk = ids[start:end].unsqueeze(0)

        with torch.inference_mode():
            logits = model(chunk).logits[0]

        first = next_token - start
        predictions = logits[first - 1:-1]
        targets = ids[next_token:end]
        total_loss += F.cross_entropy(predictions, targets, reduction="sum").item()
        total_tokens += targets.numel()
        next_token = end
    mean_loss = total_loss / total_tokens 
    perplexity = perplexity_n(torch.tensor(mean_loss))

    return {"mean_loss": mean_loss, "tokens_scored": total_tokens, "perplexity: ": perplexity}

print(evaluate_model("HuggingFaceTB/SmolLM2-135M-Instruct", "The Capital of France is Paris"))