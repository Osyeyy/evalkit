import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "HuggingFaceTB/SmolLM2-135M"
text = "The capital of france is Paris"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

ids = tokenizer(text, return_tensors="pt").input_ids

with torch.inference_mode():
    outputs = model(ids)
    logits = outputs.logits

reference_loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, logits.size(-1)),
    ids[:, 1:].reshape(-1)
)