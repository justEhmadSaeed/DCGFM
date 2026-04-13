"""
@File         :hf_download.py
@Description  :Download huggingface models and datasets from mirror site.
@Author       :Xiaojian Yuan
"""


import argparse
import os
import subprocess
import sys

# Check if huggingface_hub is installed, if not, install it
try:
    import huggingface_hub
except ImportError:
    print("Install huggingface_hub.")
    os.system("pip install -U huggingface_hub")


parser = argparse.ArgumentParser(description="HuggingFace Download Accelerator Script.")
parser.add_argument(
    "--model",
    "-M",
    default=None,
    type=str,
    help="model name in huggingface, e.g., baichuan-inc/Baichuan2-7B-Chat",
)
parser.add_argument(
    "--token",
    "-T",
    default=None,
    type=str,
    help="hugging face access token for download meta-llama/Llama-2-7b-hf, e.g., hf_***** ",
)
parser.add_argument(
    "--include",
    default=None,
    nargs="*",
    help="Glob patterns to include from files to download, e.g. --include *.json *.safetensors",
)
parser.add_argument(
    "--exclude",
    default=None,
    nargs="*",
    help="Glob patterns to exclude from files to download, e.g. --exclude *.bin *.md",
)
parser.add_argument(
    "--dataset",
    "-D",
    default=None,
    type=str,
    help="dataset name in huggingface, e.g., zh-plus/tiny-imagenet",
)
parser.add_argument(
    "--save_dir",
    "-S",
    default=None,
    type=str,
    help="path to be saved after downloading.",
)
parser.add_argument(
    "--local_dir",
    default=None,
    type=str,
    help="Explicit local directory passed to `hf download --local-dir`. Overrides the directory derived from --save_dir.",
)
parser.add_argument(
    "--repo_type",
    default="model",
    choices=["model", "dataset", "space"],
    help="Repository type for `hf download`.",
)
parser.add_argument(
    "--revision",
    default=None,
    type=str,
    help="Git revision id which can be a branch name, a tag, or a commit hash.",
)
parser.add_argument(
    "--cache_dir",
    default=None,
    type=str,
    help="Directory where to save files in the Hugging Face cache.",
)
parser.add_argument(
    "--force_download",
    action="store_true",
    help="Force download even if files are already cached.",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    help="Perform a dry run without actually downloading files.",
)
parser.add_argument(
    "--quiet",
    action="store_true",
    help="Disable progress bars and only print the download path.",
)
parser.add_argument(
    "--max_workers",
    default=None,
    type=int,
    help="Maximum number of workers to use for downloading files.",
)
parser.add_argument(
    "--use_hf_transfer", default=True, type=eval, help="Use hf-transfer, default: True"
)
parser.add_argument(
    "--use_mirror", default=True, type=eval, help="Download from mirror, default: True"
)

args = parser.parse_args()

if args.use_hf_transfer:
    # Check if hf_transfer is installed, if not, install it
    try:
        import hf_transfer
    except ImportError:
        print("Install hf_transfer.")
        os.system("pip install -U hf-transfer -i https://pypi.org/simple")
    # Enable hf-transfer if specified
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print("export HF_HUB_ENABLE_HF_TRANSFER=", os.getenv("HF_HUB_ENABLE_HF_TRANSFER"))


if args.model is None and args.dataset is None:
    print(
        "Specify the name of the model or dataset, e.g., --model baichuan-inc/Baichuan2-7B-Chat"
    )
    sys.exit()
elif args.model is not None and args.dataset is not None:
    print("Only one model or dataset can be downloaded at a time.")
    sys.exit()

if args.dataset is not None:
    args.repo_type = "dataset"

if args.use_mirror:
    # Set default endpoint to mirror site if specified
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("export HF_ENDPOINT=", os.getenv("HF_ENDPOINT"))  # https://hf-mirror.com


def build_default_local_dir(base_dir, repo_id, repo_type):
    repo_name = repo_id.split("/")
    prefix = f"{repo_type}s" if repo_type != "model" else "models"
    if len(repo_name) > 1:
        return os.path.join(base_dir, f"{prefix}--{repo_name[0]}--{repo_name[1]}")
    return os.path.join(base_dir, f"{prefix}--{repo_name[0]}")


def extend_multi_flag(cmd, flag_name, values):
    if values is None:
        return
    for value in values:
        cmd.extend([flag_name, value])


repo_id = args.model if args.model is not None else args.dataset
local_dir = args.local_dir
if local_dir is None and args.save_dir is not None:
    local_dir = build_default_local_dir(args.save_dir, repo_id, args.repo_type)

download_cmd = ["hf", "download", repo_id, "--repo-type", args.repo_type]

if args.revision is not None:
    download_cmd.extend(["--revision", args.revision])
if args.cache_dir is not None:
    download_cmd.extend(["--cache-dir", args.cache_dir])
if local_dir is not None:
    download_cmd.extend(["--local-dir", local_dir])
if args.token is not None:
    download_cmd.extend(["--token", args.token])
if args.force_download:
    download_cmd.append("--force-download")
if args.dry_run:
    download_cmd.append("--dry-run")
if args.quiet:
    download_cmd.append("--quiet")
if args.max_workers is not None:
    download_cmd.extend(["--max-workers", str(args.max_workers)])

extend_multi_flag(download_cmd, "--include", args.include)
extend_multi_flag(download_cmd, "--exclude", args.exclude)

raise_code = subprocess.run(download_cmd, check=False).returncode
sys.exit(raise_code)
