# Running OFA Molecular Graphs on Google Colab

This guide explains how to run the OFA part of this repository on Google Colab for the molecular graph workflow.

Important limitation:

- The current `OFA/run_cdm.py` pruning pipeline assumes **three training tasks**.
- The provided config `OFA/yamls/soft_and_hard.yaml` uses:
  - `arxiv_fs`
  - `FB15K237_fs`
  - `mol_fs`
- `mol_fs` is the molecular graph task, but it is currently wired into that three-task setup.
- In other words, the existing Colab recipe below runs the repo's current OFA pipeline that includes the molecular task, not a molecule-only hard/soft pruning run.

If you only want a molecule-only OFA run, `run_cdm.py` needs code changes first.

## 1. Start a GPU Colab Runtime

In Colab:

1. Open `Runtime -> Change runtime type`
2. Set `Hardware accelerator` to `GPU`
3. Save

Then verify the GPU:

```bash
!nvidia-smi
```

## 2. Clone the Repository

```bash
%cd /content
!git clone https://github.com/justEhmadSaeed/DCGFM.git
%cd /content/DCGFM/OFA
```

If you already uploaded the repository to Drive instead of cloning it, just `cd` into the `OFA/` directory.

## 3. Install Dependencies

The repository ships a Conda environment for local machines, but Colab is easier to run with `pip`.

Run this in one cell:

```bash
!pip install -q --upgrade pip
!pip install -q -r /content/DCGFM/requirements.txt
```

Notes:

- `deepspeed` is required because it is imported by the OFA utility layer even when you only use one GPU.
- `rdkit-pypi` is required because the molecule loader converts SMILES strings into graphs.
- `wandb` is required because `run_cdm.py` creates a `WandbLogger`.

## 3.1 Install PyG Extension Wheels

This repository imports `torch_scatter` directly, so `torch-geometric` alone is not enough on Colab.

Run this after installing `requirements.txt`:

```python
import os
import torch

torch_version = torch.__version__.split("+")[0]
cuda_version = f"cu{torch.version.cuda.replace('.', '')}"
wheel_index = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html"

print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("PyG wheel index:", wheel_index)

os.system(
    f"pip install -q pyg_lib torch_scatter torch_sparse torch_cluster -f {wheel_index}"
)
```

Quick verification:

```python
import torch_scatter, torch_sparse, torch_cluster
print("PyG extension wheels installed correctly.")
```

## 4. Download the Required Model and Dataset

The molecule code expects these exact local cache paths:

- model: `OFA/cache_data/model/models--sentence-transformers--multi-qa-distilbert-cos-v1`
- dataset: `OFA/cache_data/dataset/datasets--haitengzhao--molecule_property_instruction`

The repo already includes a downloader script. Run:

```bash
%cd /content/DCGFM/OFA

!python hf/hf_download.py \
  --model sentence-transformers/multi-qa-distilbert-cos-v1 \
  --save_dir ./cache_data/model \
  --quiet \
  --use_mirror False

!python hf/hf_download.py \
  --dataset haitengzhao/molecule_property_instruction \
  --save_dir ./cache_data/dataset \
  --quiet \
  --use_mirror False
```

The helper script now wraps the current `hf download` CLI. If you need more control, it also accepts current download options such as:

- `--revision`
- `--cache_dir`
- `--local_dir`
- `--include`
- `--exclude`
- `--force_download`
- `--dry_run`
- `--quiet`
- `--max_workers`

Optional quick check:

```bash
!ls ./cache_data/model
!ls ./cache_data/dataset
```

You should see directories named:

- `models--sentence-transformers--multi-qa-distilbert-cos-v1`
- `datasets--haitengzhao--molecule_property_instruction`

## 5. Avoid WandB Login Prompts in Colab

Run:

```bash
%env WANDB_MODE=offline
```

This keeps logging local and avoids interactive login prompts.

## 6. Run a Small Smoke Test First

The checked-in `yamls/soft_and_hard.yaml` has been reduced for lighter Colab runs:

- `fs_sample_size: 12000` which is 20% of the original `60000`
- `num_epochs: 5` instead of `30`
- `hard_pruning_epochs: 5` instead of `25`

Start with an even smaller smoke test to verify the environment.

This uses the repo's three-task setup from `yamls/soft_and_hard.yaml`, but reduces the workload:

```bash
%cd /content/DCGFM/OFA

!CUDA_VISIBLE_DEVICES=0 python run_cdm.py \
  --control_gpu \
  --gpus 0 \
  --override yamls/soft_and_hard.yaml \
  --hard_pruning_mode hard_prune_api \
  --hard_pruning_joint \
  --hard_pruning_reverse \
  --hard_pruning_ratio 0.3 \
  --prune_ratio 0.3 \
  --fs_sample_size 2000 \
  --checkpoint_interval 1 \
  num_epochs 2 \
  hard_pruning_epochs 2 \
  batch_size 32 \
  eval_batch_size 32 \
  offline_log true
```

What this does:

- keeps the molecular task `mol_fs` in the run
- keeps the original pruning path
- cuts epochs, batch size, and sample size to fit Colab better

## 7. Run the Full Configuration

Once the smoke test works, run the OFA command in `README.md` with GPU auto-detection.
`README.md` lists `0.3/0.5/0.7`; in Colab, run that as three separate runs:

```python
%cd /content/DCGFM/OFA

import os
import torch

gpu_ids = ",".join(str(i) for i in range(torch.cuda.device_count()))
if not gpu_ids:
    raise RuntimeError("No GPU detected. Switch Colab runtime to GPU first.")

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
print("Using GPUs:", gpu_ids)

!python run_cdm.py \
  --control_gpu \
  --gpus {gpu_ids} \
  --save_model \
  --override yamls/soft_and_hard.yaml \
  --hard_pruning_mode hard_prune_api \
  --hard_pruning_joint \
  --hard_pruning_reverse \
  --hard_pruning_ratio 0.3 \
  --prune_ratio 0.3 \
  offline_log true
```

This is the runnable Colab equivalent of the OFA command in `README.md`, while using all available GPUs.

## 8. Where Outputs Go

The run writes artifacts under:

- `OFA/saved_exp/`
- `OFA/cache_data/ofa_dataset/`
- `OFA/lightning_logs/`

If you want outputs to persist after the Colab session ends, mount Google Drive and run from there.

Example:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then clone or copy the repository into Drive and run from that location.

## 9. What the Molecular Part Actually Is

The molecular task in this setup is `mol_fs`.

From the config files:

- training dataset: `chemblpre_fs`
- evaluation datasets:
  - `chemhiv_fs_20`
  - `chempcba_fs_20`

So the molecular portion is a few-shot molecular graph setup, not a plain single-dataset classification run.

## 10. If You Want a Molecule-Only Colab Workflow

That is not fully supported by the current `run_cdm.py` pruning path.

Specifically:

- `get_effective_indices()` in `run_cdm.py` is hard-coded around three train datasets
- the provided OFA pruning config is built around `arxiv_fs`, `FB15K237_fs`, and `mol_fs`

If you want, the next step is to refactor `run_cdm.py` so the hard-pruning path works with an arbitrary number of tasks, including a molecule-only run on Colab.
