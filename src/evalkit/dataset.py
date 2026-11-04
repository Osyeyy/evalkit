# download dataset fresh samples / choosen one with seed if needed (for comparison)
from datasets import load_dataset

DATASETS = {
    "wikitext2": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "split": "test"

    },
    "wikitext103": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-103-raw-v1",
        "split": "test"
    }
}

def load_text_dataset(dataset_id, max_samples=None, seed=42):
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_id}")

    config = DATASETS[dataset_id]

    dataset = load_dataset(
        config["path"],
        config["name"],
        split=config["split"]
    )

    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(
            range(min(max_samples, len(dataset)))

        )

    text = "\n\n".join(dataset["text"])

    return text