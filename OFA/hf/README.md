# 国内用户 HuggingFace 高速下载

利用 Hugging Face 官方的 [`hf download`](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-download) 命令和 [hf_transfer](https://github.com/huggingface/hf_transfer) 从 [HuggingFace 镜像站](https://hf-mirror.com/)上对模型和数据集进行高速下载。

---
脚本 `hf_download.py` 现在封装的是新版 `hf download` 接口，而不是旧的 `huggingface-cli download`。

- 下载指定的文件: `--include tokenizer.model tokenizer_config.json`
- 下载某一类文件: `--include "*.bin"`
- 不下载指定文件: `--exclude "*.md"`
- 也可以同时使用: `--include "*.json" --exclude "config.json"`

## Usage

### 下载模型

从HuggingFace上获取到所需模型名，例如 `lmsys/vicuna-7b-v1.5`：

```bash
python hf_download.py --model lmsys/vicuna-7b-v1.5 --save_dir ./hf_hub
```
如果下载需要授权的模型，例如 meta-llama 系列，则需要指定 `--token` 参数为你的 Hugging Face Access Token。

也可以传递新版 CLI 的常见参数，例如：

```bash
python hf_download.py \
  --model lmsys/vicuna-7b-v1.5 \
  --save_dir ./hf_hub \
  --revision main \
  --include "*.json" "*.safetensors" \
  --exclude "*.md" \
  --quiet
```

**注意事项：**

（1）脚本内置通过 pip 自动安装 `huggingface_hub` 和 `hf_transfer`。如果 `hf_transfer` 版本低于 0.1.4 则不会显示下载进度条，可以手动更新：
```
pip install -U hf-transfer -i https://pypi.org/simple
```
如出现 `hf: command not found` 或其他 CLI 问题，尝试重新安装：
```
pip install -U huggingface_hub
```
如出现关于 `hf_transfer` 的报错，可以通过 `--use_hf_transfer False` 参数关闭 hf_transfer。

（2）若指定了 `save_dir`，脚本会为 `hf download --local-dir` 自动生成与仓库兼容的目录结构，例如：

- 模型：`models--org--name`
- 数据集：`datasets--org--name`

例如：
```python
from transformers import pipeline
pipe = pipeline("text-generation", model="./hf_hub/models--lmsys--vicuna-7b-v1.5")
```
若不指定 `save_dir` 或 `local_dir`，则会下载到默认缓存目录 `~/.cache/huggingface/hub` 中，这时调用模型可以直接使用模型名称 `lmsys/vicuna-7b-v1.5`。

（3）若不想在调用时使用绝对路径，又不希望将所有模型保存在默认路径下，可以通过**软链接**的方式进行设置，步骤如下：
- 先在任意位置创建目录，作为下载文件的真实存储位置，例如：
    ```bash
    mkdir /data/huggingface_cache
    ```
- 若 transforms 已经在默认位置 `~/.cache/huggingface/hub` 创建了目录，需要先删除：
    ```bash
    rm -r ~/.cache/huggingface
    ```
- 创建软链接指向真实存储目录：
    ```bash
    ln -s /data/huggingface_cache ~/.cache/huggingface
    ``` 
- 之后运行下载脚本时**不要指定** `save_dir`，会自动下载至第一步创建的目录下：
    ```bash
    python hf_download.py --model lmsys/vicuna-7b-v1.5
    ```
- 通过这种方式，调用模型时可以直接使用模型名称，而不需要使用存储路径：
    ```bash
    from transformers import pipeline
    pipe = pipeline("text-generation", model="lmsys/vicuna-7b-v1.5")
    ```

### 下载数据集

和下载模型同理，以 `zh-plus/tiny-imagenet` 为例:
```bash
python hf_download.py --dataset zh-plus/tiny-imagenet --save_dir ./hf_hub
```

也支持新版 `hf download` 的数据集相关参数：

```bash
python hf_download.py \
  --dataset zh-plus/tiny-imagenet \
  --save_dir ./hf_hub \
  --revision main \
  --include "*.json" "*.parquet"
```

### 参数说明
 - `--model`: Hugging Face 上要下载的模型名称，例如 `--model lmsys/vicuna-7b-v1.5`
 - `--dataset`: Hugging Face 上要下载的数据集名称，例如 `--dataset zh-plus/tiny-imagenet`
 - `--save_dir`: 下载后自动生成兼容目录结构的保存根目录
 - `--local_dir`: 直接传给 `hf download --local-dir` 的目标目录，优先级高于 `--save_dir`
 - `--repo_type`: 资源类型，支持 `model`、`dataset`、`space`
 - `--revision`: 分支名、tag 或 commit hash
 - `--cache_dir`: Hugging Face 缓存目录
 - `--token`: 下载需要登录的模型或数据时使用的 token，格式为 `hf_****`
 - `--include`: 指定要下载的文件模式，例如 `--include "*.json" "*.safetensors"`
 - `--exclude`: 指定不要下载的文件模式，例如 `--exclude "*.md"`
 - `--force_download`: 即使本地已缓存也强制重新下载
 - `--dry_run`: 仅预览，不实际下载
 - `--quiet`: 关闭进度条，只输出最终路径
 - `--max_workers`: 设置并发下载 worker 数量
 - `--use_hf_transfer`: 使用 hf-transfer 进行加速下载，默认开启
 - `--use_mirror`: 从镜像站 `https://hf-mirror.com/` 下载，默认开启，国内用户建议开启
