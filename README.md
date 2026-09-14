# Agentic Communication R1

This repository contains the code for the thesis project:

**Learnable Reading Policies for Multi-Role Collaborative Reasoning in a Shared Latent Space**

The project studies selective latent KV-cache reading in a shared-backbone multi-role reasoning system. The implementation uses a frozen Qwen3-4B backbone and trains only a lightweight Reading Policy for selecting upstream latent KV chunks.

## Important: Model Weights Are Not Included

This repository does **not** include the Qwen3-4B model weights.

The model checkpoint is large and should not be committed to GitHub. To run the full system, please download the required model separately and place it under the expected local directory.

The default expected model path used by this project is:

```bash
models/Qwen3-4B
```

After setup, the repository should look like this:

```text
Agentic-Communication-R1/
├── models/
│   └── Qwen3-4B/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── model-00001-of-00003.safetensors
│       ├── model-00002-of-00003.safetensors
│       ├── model-00003-of-00003.safetensors
│       └── ...
├── scripts/
├── agent_system/
├── data/
└── README.md
```

The exact file names may differ slightly depending on the model release, but the directory must contain a Hugging Face / Transformers-compatible Qwen3-4B checkpoint.

## Downloading Qwen3-4B

The project expects the Hugging Face model repository:

```text
Qwen/Qwen3-4B
```

You can download the model with either `huggingface-cli` or Python.

### Option 1: Download with huggingface-cli

Install the Hugging Face Hub client:

```bash
pip install -U huggingface_hub
```

Create the local model directory:

```bash
mkdir -p models
```

Download the model into the expected path:

```bash
huggingface-cli download Qwen/Qwen3-4B \
  --local-dir models/Qwen3-4B \
  --local-dir-use-symlinks False
```

After the download finishes, check that the directory exists:

```bash
ls models/Qwen3-4B
```

### Option 2: Download with Python

You can also download the model with Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    local_dir="models/Qwen3-4B",
    local_dir_use_symlinks=False,
)
```

### Option 3: Use an Existing Local Model Path

If you already have Qwen3-4B stored elsewhere, you can either copy or symlink it:

```bash
mkdir -p models
ln -s /path/to/your/Qwen3-4B models/Qwen3-4B
```

Alternatively, modify the model path in the relevant configuration or run script, for example:

```bash
--model_path /path/to/your/Qwen3-4B
```

## Recommended Environment

Install the core Python dependencies:

```bash
pip install -U torch transformers accelerate safetensors huggingface_hub
```

For Qwen3 models, please use a recent version of `transformers`. Older versions may not support the model configuration correctly.

## Quick Model Check

After downloading the model, you can verify that it can be loaded with the following script:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

print("Qwen3-4B loaded successfully.")
```

If this script runs without errors, the model path is correctly configured.

## Repository Structure

A typical local setup should follow this structure:

```text
Agentic-Communication-R1/
├── agent_system/
│   └── ...
├── scripts/
│   └── ...
├── data/
│   └── ...
├── models/
│   └── Qwen3-4B/
│       └── ...
├── README.md
└── .gitignore
```

The `models/` directory is intended for local use only and should not be pushed to GitHub.

## Git Ignore Recommendation

Please make sure that large local checkpoints are excluded from version control. A recommended `.gitignore` entry is:

```gitignore
# Local model checkpoints
models/
*.safetensors
*.bin
*.pt
*.pth
*.ckpt

# Hugging Face cache
.cache/
huggingface/

# Python cache
__pycache__/
*.pyc
```

If you want to keep the `models/` directory visible in GitHub while still excluding model files, use:

```bash
mkdir -p models
touch models/.gitkeep
```

Then use this `.gitignore` rule instead:

```gitignore
models/*
!models/.gitkeep
```

This keeps the folder structure while preventing large model checkpoints from being committed.

## Notes

- The Qwen3-4B model weights are not distributed with this repository.
- Users must download the model separately before running the full system.
- Users are responsible for following the license and usage terms of the downloaded model.
- The experiments in the thesis were conducted with a frozen Qwen3-4B backbone.
- Changing the backbone may affect the behavior of the Reading Policy and may lead to results different from those reported in the thesis.
