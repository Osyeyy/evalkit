#  find huggingface model, find expected files, download to workspace 

import os
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
expected_files = [
    "config.json",
    "adapter_config.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "generation_config.json"
]


model_id = input("Which model do you want to check?").strip()

try:
    download = snapshot_download(repo_id=model_id, local_dir=f"./workspace/{model_id}", allow_patterns=expected_files, repo_type="model")
except GatedRepoError:
    raise SystemExit(f"Repo '{model_id}' requires license acceptation / hf token.")
except RepositoryNotFoundError:
    raise SystemExit(f"Repo '{model_id}' does not exists ( or it is private).")


def verify_download(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def check_files(path):
    downloaded = verify_download(path)
    informations = {}
    for name in expected_files:
        if name in downloaded:
            print(f"ok     {name}")
            informations[name] = os.path.join(path, name)
        else: 
            print(f"missing {name}")
            informations[name] = None
    return informations
print(verify_download(download))
files_info = check_files(download)
print(files_info)