# download dataset fresh samples / choosen one with seed if needed (for comparison)
from datasets import load_dataset

def load_wikitext():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    text = "\n\n".join(dataset["text"])
    return text

