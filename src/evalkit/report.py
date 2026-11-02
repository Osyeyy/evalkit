# save json report to not loose results after dc 

import json 
from pathlib import Path

def save_report(model_id, dataset_name, results):
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_id.replace("/", "_")
    output_path = output_dir / f"{safe_name}.json"

    report = {
        "model_id": model_id,
        "dataset": dataset_name,
        "evaluation": {
            "mean_loss": results["mean_loss"],
            "perplexity": results["perplexity"],
            "tokens_scored": results["tokens_scored"]
        },
        "settings": {
            "window": results["window"],
            "context": results["context"]
        }
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return output_path

