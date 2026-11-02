### tokenization, forward pass, results 

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from evalkit.metrics import perplexity_n
from tqdm import tqdm

def evaluate_model(model_id, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    window = 512 
    context = 384 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(ids) < 2: 
        raise ValueError("Text must have at least 2 tokens")

    total_loss = 0.0
    total_tokens = 0
    next_token = 1 

    progress = tqdm(total=len(ids) - 1, desc="Evaluating", unit="tok")

    
    while next_token < len(ids):
        start = max(0, next_token - context)
        end = min(start + window, len(ids))
        chunk = ids[start:end].unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(chunk).logits[0]

        first = next_token - start
        predictions = logits[first - 1:-1]
        targets = ids[next_token:end].to(device)
        total_loss += F.cross_entropy(predictions, targets, reduction="sum").item()
        total_tokens += targets.numel()
        current_mean_loss = total_loss / total_tokens 
        progress.update(targets.numel())
        progress.set_postfix(mean_loss=f"{current_mean_loss:.4f}")
        next_token = end
    mean_loss = total_loss / total_tokens 
    perplexity = perplexity_n(torch.tensor(mean_loss))
    progress.close()

    expected_tokens = len(ids) - 1
    assert total_tokens == expected_tokens, (
        f"token mismatch: {total_tokens} != {expected_tokens}"
    )

    return {"mean_loss": mean_loss, "tokens_scored": total_tokens, "perplexity": perplexity, "window": window, "context": context}
