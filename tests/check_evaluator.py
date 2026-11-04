import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM


from evalkit.evaluator import evaluate_model 

def test_sliding_window():
    model_id ="HuggingFaceTB/SmolLM2-135M"
    text = "The capital of France is Paris. This is a test of language modeling. " * 150

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ids = tokenizer(text, return_tensors="pt").input_ids

    assert ids.size(1) > 1024

    result = evaluate_model(model_id, text)

    assert result["tokens_scored"] == ids.size(1) - 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    window = 512
    context = 384
    next_token = 1
    total_loss = 0.0
    total_tokens = 0

    while next_token < ids.size(1):
        start = max(0, next_token - context)
        end = min(start + window, ids.size(1))

        chunk = ids[:, start:end].to(device)
        labels = chunk.clone()

        first = next_token - start
        labels[:, :first] = -100

        with torch.inference_mode():
            outputs = model(chunk, labels=labels)

        scored = end - next_token
        total_loss += outputs.loss.item() * scored
        total_tokens += scored
        next_token = end

    reference_loss = total_loss / total_tokens

    print(f"evalkit loss : {result['mean_loss']:.6f}")
    print(f"reference loss : {reference_loss:.6f}")
    print(f"Tokens scored: {result['tokens_scored']}")

    assert total_tokens == ids.size(1) - 1
    assert result["mean_loss"] == pytest.approx(reference_loss, abs=1e-3)

    print("Sliding window test passed")


if __name__ == "__main__":
    test_sliding_window()