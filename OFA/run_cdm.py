import argparse
import os
from types import SimpleNamespace

import torch
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import AUROC, Accuracy

import utils
from gp.lightning.data_template import DataModule
from gp.lightning.metric import (
    flat_binary_func,
    EvalKit,
)
from gp.lightning.module_template import ExpConfig
from gp.lightning.training import lightning_fit, lightning_test
from gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
)

from lightning_model import IBGraphPredLightning
from models.model import BinGraphModel, BinGraphAttModel
from models.model import PyGRGCNEdge
from task_constructor import UnifiedTaskConstructor
from utils import (
    SentenceEncoder,
    MultiApr,
    MultiAuc,
)
from hard_prune_module import hard_prune_api
import numpy as np
import torch_geometric as pyg
import pickle

# ADDED: imports for embedding saving and ACF plotting
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for server / HPC use
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.decomposition import PCA

# ADDED: directory where embeddings (.npy) and ACF plots (.png) are written.
# Override this variable to redirect output without touching the rest of the code.
EMBEDDINGS_SAVE_DIR = "saved_embeddings"

# ---------------------------------------------------------------------------
# ADDED: Embedding-saving and ACF-plotting helpers
# ---------------------------------------------------------------------------

def _save_embeddings_and_acf(embeddings, dataset_name, stage, save_dir):
    """Save embeddings as .npy and produce an ACF plot for the given stage.

    Args:
        embeddings (np.ndarray): Shape (N, D) — one row per subgraph.
        dataset_name (str):      Human-readable name used in filenames/titles.
        stage (str):             "pre_prune" or "post_prune".
        save_dir (str):          Directory to write files into (created if needed).

    Dimensionality-reduction strategy for ACF:
        Embeddings are reduced to a scalar per sample via PCA (1 component).
        PCA is preferred over a plain mean because it projects along the axis of
        maximum variance, giving a more informative 1-D signal.  The resulting
        sequence is treated as a 1-D time series ordered by sample index, and
        the ACF is computed over that sequence.
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- save raw embeddings ---
    emb_filename = f"{dataset_name}_embeddings_{stage}.npy"
    emb_path = os.path.join(save_dir, emb_filename)
    np.save(emb_path, embeddings)
    print(f"[DCGFM] Saved embeddings  → {emb_path}  shape={embeddings.shape}")

    # --- reduce to 1-D for ACF (PCA → 1 component) ---
    n_samples = embeddings.shape[0]
    if n_samples < 4:
        print(f"[DCGFM] Skipping ACF for {dataset_name}/{stage}: too few samples ({n_samples}).")
        return

    if embeddings.shape[1] > 1:
        pca = PCA(n_components=1, random_state=0)
        signal_1d = pca.fit_transform(embeddings).ravel()
    else:
        signal_1d = embeddings.ravel()

    # number of lags: up to 40 or half the series length − 1, whichever is smaller
    max_lags = min(40, n_samples // 2 - 1)
    if max_lags < 1:
        print(f"[DCGFM] Skipping ACF for {dataset_name}/{stage}: series too short for lags.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(signal_1d, ax=ax, lags=max_lags, alpha=0.05)
    ax.set_title(f"{dataset_name} — {stage.replace('_', ' ').title()} — ACF Plot")
    ax.set_xlabel("Lag (subgraph index)")
    ax.set_ylabel("Autocorrelation")

    plot_filename = f"{dataset_name}_acf_{stage}.png"
    plot_path = os.path.join(save_dir, plot_filename)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DCGFM] Saved ACF plot    → {plot_path}")

    # -----------------------------------------------------------------------
    # EXTENSION SECTION — add extra plots using the saved embedding data here.
    # -----------------------------------------------------------------------
    # Example: compute and plot the histogram of embedding norms
    #
    #   norms = np.linalg.norm(embeddings, axis=1)
    #   fig2, ax2 = plt.subplots()
    #   ax2.hist(norms, bins=50)
    #   ax2.set_title(f"{dataset_name} — {stage} — Embedding Norm Histogram")
    #   fig2.savefig(os.path.join(save_dir, f"{dataset_name}_norm_hist_{stage}.png"), dpi=150)
    #   plt.close(fig2)
    #
    # Example: 2-D PCA scatter coloured by subgraph index
    #
    #   pca2d = PCA(n_components=2, random_state=0).fit_transform(embeddings)
    #   fig3, ax3 = plt.subplots()
    #   sc = ax3.scatter(pca2d[:, 0], pca2d[:, 1], c=np.arange(len(pca2d)), cmap="viridis", s=5)
    #   plt.colorbar(sc, ax=ax3, label="sample index")
    #   ax3.set_title(f"{dataset_name} — {stage} — PCA 2-D")
    #   fig3.savefig(os.path.join(save_dir, f"{dataset_name}_pca2d_{stage}.png"), dpi=150)
    #   plt.close(fig3)
    # -----------------------------------------------------------------------


def get_useful_indices(data_list_no_prompt, hard_pruning_mode, hard_pruning_ratio, hard_pruning_reverse, hard_pruning_epochs,
                       emb_cache=None):
    # MODIFIED: added optional emb_cache parameter (default None keeps existing behaviour).
    # When a list is passed, it is forwarded to the underlying pruning function so that
    # per-graph embeddings are collected as a side-effect of the normal scoring pass.
    # Only pass emb_cache when the called function accepts it (i.e. hard_prune_api).
    call_kwargs = {"data_list": data_list_no_prompt, "max_epochs": hard_pruning_epochs}
    if emb_cache is not None:
        call_kwargs["emb_cache"] = emb_cache
    anomany_scores = globals()[hard_pruning_mode](**call_kwargs)
    if hard_pruning_reverse == False: 
        threshold = np.percentile(anomany_scores, (1.0 - hard_pruning_ratio) * 100)
        useful_indices = np.where(anomany_scores <= threshold)[0].tolist()
    else: 
        threshold = np.percentile(anomany_scores, hard_pruning_ratio * 100)
        useful_indices = np.where(anomany_scores >= threshold)[0].tolist() 
    return useful_indices

def get_effective_indices(params, tasks):
    indices_path_mode = f"dcgfm_{params.hard_pruning_mode}_{params.hard_pruning_epochs}_{params.hard_pruning_ratio}"
    effective_indices = [[] for i in range(3)] 
    if params.hard_pruning_ratio == 0.0:
        for i in range(3):
            effective_indices[i] = list(range(params.fs_sample_size))
        return effective_indices
    
    data_list_no_prompt, data_list_no_prompt_ind, right_bound = [], [], []
    data_list_no_link = []
    for i in range(3):
        data_list_no_prompt_ind.append(tasks.datasets["train"][i].all_no_prompt_data)
        right_bound.append((i + 1) * params.fs_sample_size) 
        if i != 1: 
            data_list_no_link.extend(tasks.datasets["train"][i].all_no_prompt_data)
    indices_dir = os.path.join(params.big_data_cache_dir, "hard_pruning_indices")
    if not os.path.exists(indices_dir):
        os.makedirs(indices_dir, exist_ok=True)
    indices_path_0 = os.path.join(indices_dir, f"{indices_path_mode}_0.pkl")
    if not os.path.exists(indices_path_0): 
        if params.hard_pruning_mode == "random": 
            useful_indices_all = np.sort(np.random.choice(
                                    params.fs_sample_size * 3, 
                                    size=int(params.fs_sample_size * 3 * (1.0 - params.hard_pruning_ratio)), 
                                    replace=False
                                )).tolist()             
            for i in range(len(useful_indices_all)):
                if useful_indices_all[i] < right_bound[0]:
                    effective_indices[0].append(useful_indices_all[i])
                elif useful_indices_all[i] < right_bound[1]:
                    effective_indices[1].append(useful_indices_all[i] - right_bound[0])
                else:
                    effective_indices[2].append(useful_indices_all[i] - right_bound[1])
        else:
            # ADDED: derive human-readable dataset type names for file naming.
            # Index 0 = node-level, 1 = link-level, 2 = graph-level tasks (OFA convention).
            task_type_names = ["node", "link", "graph"]

            if params.hard_pruning_joint == True:
                # Joint mode: node+graph data pruned together; link is random.
                # Embeddings for the combined list are saved under "joint" names.
                emb_cache_joint = []  # ADDED: Point A cache for joint data
                useful_indices_all = get_useful_indices(
                    data_list_no_link, params.hard_pruning_mode,
                    params.hard_pruning_ratio, params.hard_pruning_reverse,
                    params.hard_pruning_epochs, emb_cache=emb_cache_joint,
                )
                # ADDED: Point A — save pre-prune embeddings for joint (node+graph) data
                if emb_cache_joint:
                    pre_embs_joint = np.array(emb_cache_joint)
                    _save_embeddings_and_acf(pre_embs_joint, "joint_node_graph", "pre_prune", EMBEDDINGS_SAVE_DIR)
                    # ADDED: Point B — save post-prune embeddings (only the kept indices)
                    post_embs_joint = pre_embs_joint[useful_indices_all]
                    _save_embeddings_and_acf(post_embs_joint, "joint_node_graph", "post_prune", EMBEDDINGS_SAVE_DIR)

                effective_indices[1] = np.sort(np.random.choice(
                                            params.fs_sample_size,
                                            size=int(params.fs_sample_size * (1.0 - params.hard_pruning_ratio)),
                                            replace=False
                                        )).tolist()
                for i in range(len(useful_indices_all)):
                    if useful_indices_all[i] < right_bound[0]:
                        effective_indices[0].append(useful_indices_all[i])
                    else:
                        effective_indices[2].append(useful_indices_all[i] - right_bound[0])
            else:
                for i in range(3):
                    # ADDED: Point A — collect per-graph embeddings before pruning threshold is applied
                    emb_cache_i = []
                    effective_indices[i] = get_useful_indices(
                        data_list_no_prompt_ind[i], params.hard_pruning_mode,
                        params.hard_pruning_ratio, params.hard_pruning_reverse,
                        params.hard_pruning_epochs, emb_cache=emb_cache_i,
                    )
                    # ADDED: save Point A embeddings (all subgraphs, pre-prune)
                    if emb_cache_i:
                        pre_embs = np.array(emb_cache_i)
                        dataset_label = task_type_names[i]
                        _save_embeddings_and_acf(pre_embs, dataset_label, "pre_prune", EMBEDDINGS_SAVE_DIR)
                        # ADDED: Point B — select only the kept indices for post-prune embeddings
                        post_embs = pre_embs[effective_indices[i]]
                        _save_embeddings_and_acf(post_embs, dataset_label, "post_prune", EMBEDDINGS_SAVE_DIR)
            for i in range(3):
                effective_indices[i] = np.sort(effective_indices[i]).tolist()
        
        for i in range(3):
            pkl_file_name = os.path.join(indices_dir, f"{indices_path_mode}_{i}.pkl")
            with open(pkl_file_name, "wb") as f:
                pickle.dump(effective_indices[i], f)
        
    else: 
        for i in range(3):
            file_name = os.path.join(indices_dir, f"{indices_path_mode}_{i}.pkl")
            with open(file_name, "rb") as f:
                effective_indices[i] = pickle.load(f)
    return effective_indices

def main(params):
    """
    0. Check GPU setting.
    """
    if params.control_gpu:
        gpu_ids = params.gpus
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device, gpu_ids = utils.get_available_devices()
    gpu_size = len(gpu_ids)
    print(f"Using GPU: {gpu_ids}")
    print(f"GPU size: {gpu_size}")

    """
    1. Initiate task constructor.
    """
    encoder = SentenceEncoder(params.llm_name, batch_size=params.llm_b_size)

    task_config_lookup = load_yaml(
        os.path.join(os.path.dirname(__file__), "configs", "task_config.yaml")
    )
    data_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__), "configs", "data_config.yaml"))

    if isinstance(params.task_names, str):
        task_names = [a.strip() for a in params.task_names.split(",")]
    else:
        task_names = params.task_names

    tasks = UnifiedTaskConstructor(
        task_names,
        params.load_texts,
        encoder,
        task_config_lookup,
        data_config_lookup,
        batch_size=params.batch_size,
        sample_size=params.train_sample_size,
        num_epochs=params.num_epochs,
        prune_ratio=params.prune_ratio,
        delta=params.delta,
        fs_sample_size=params.fs_sample_size,
        hard_pruning_ratio=params.hard_pruning_ratio,
        hard_pruning_epochs=params.hard_pruning_epochs,
    )
    val_task_index_lst, val_pool_mode = tasks.construct_exp()

    """
    1.5 make hard pruning
    """
    current_effective_indices = get_effective_indices(params, tasks)
    
    for i in range(3):
        tasks.datasets["train"][i].effective_indices = current_effective_indices[i]
    
    if encoder is not None:
        encoder.flush_model()
    
    """
    2. Load model 
    """
    out_dim = params.emb_dim + (params.rwpe if params.rwpe is not None else 0)

    gnn = PyGRGCNEdge(
        params.num_layers,
        5,
        out_dim,
        out_dim,
        drop_ratio=params.dropout,
        JK=params.JK,
    )

    bin_model = BinGraphAttModel if params.JK == "none" else BinGraphModel
    model = bin_model(model=gnn, llm_name=params.llm_name, outdim=out_dim, task_dim=1,
                      add_rwpe=params.rwpe, dropout=params.dropout)

    """
    3. Construct datasets and lightning datamodule.
    """

    if hasattr(params, "d_multiple"):
        if isinstance(params.d_multiple, str):
            data_multiple = [float(a) for a in params.d_multiple.split(",")]
        else:
            data_multiple = params.d_multiple
    else:
        data_multiple = [1]

    if hasattr(params, "d_min_ratio"):
        if isinstance(params.d_min_ratio, str):
            min_ratio = [float(a) for a in params.d_min_ratio.split(",")]
        else:
            min_ratio = params.d_min_ratio
    else:
        min_ratio = [1]
    train_data = tasks.make_train_data()
    text_dataset = tasks.make_full_dm_list()
    params.datamodule = DataModule(
        text_dataset, gpu_size=gpu_size, num_workers=params.num_workers,
        num_epochs=params.num_epochs, prune_ratio=params.prune_ratio, delta=params.delta
    )

    """
    4. Initiate evaluation kit. 
    """
    eval_data = text_dataset["val"] + text_dataset["test"]
    val_state = [dt.state_name for dt in text_dataset["val"]]
    test_state = [dt.state_name for dt in text_dataset["test"]]
    eval_state = val_state + test_state
    eval_metric = [dt.metric for dt in eval_data]
    eval_funcs = [dt.meta_data["eval_func"] for dt in eval_data]
    
    loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    evlter = []
    for dt in eval_data:
        if dt.metric == "acc":
            evlter.append(Accuracy(task="multiclass", num_classes=dt.classes))
        elif dt.metric == "auc":
            evlter.append(AUROC(task="binary"))
        elif dt.metric == "apr":
            evlter.append(MultiApr(num_labels=dt.classes))
        elif dt.metric == "aucmulti":
            evlter.append(MultiAuc(num_labels=dt.classes))

    metrics = EvalKit(
        eval_metric,
        evlter,
        loss,
        eval_funcs,
        flat_binary_func,
        eval_mode="max",
        exp_prefix="",
        eval_state=eval_state,
        test_monitor_state=test_state[0],
    )

    """
    5. Initiate optimizer, scheduler and lightning model module.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, weight_decay=params.l2
    )
    lr_scheduler = {
        "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 15, 0.5),
        "interval": "epoch",
        "frequency": 1,
    }

    
    exp_config = ExpConfig(
        "",
        optimizer,
        
        lr_scheduler=lr_scheduler,
    )
    exp_config.val_state_name = val_state
    exp_config.test_state_name = test_state

    pred_model = IBGraphPredLightning(exp_config, model, metrics, prune_ratio=params.prune_ratio, delta=params.delta)

    """
    6. Start training and logging.
    """
    wandb_logger = WandbLogger(
        project=params.log_project,
        name=params.exp_name,
        save_dir=params.exp_dir,
        offline=params.offline_log,
    )
    strategy = "deepspeed_stage_2" if gpu_size > 1 else "auto"
    
    test_res = lightning_fit(
        wandb_logger,
        pred_model,
        params.datamodule,
        metrics,
        params.num_epochs,
        strategy=strategy,
        save_model=params.save_model,
        load_best=params.load_best,
        reload_freq=params.reload_freq,
        test_rep=params.test_rep,
        val_interval=params.val_interval,
        limit_val_batches=params.limit_val_batches,
        cktp_prefix=params.big_data_cache_dir + "/" + params.exp_name,
        checkpoint_interval=params.checkpoint_interval,
        fs_sample_size=params.fs_sample_size,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rl")
    parser.add_argument("--override", type=str)

    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )

    parser.add_argument("--control_gpu", action="store_true", help="Control GPU usage")
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated list of GPU IDs to use")

    parser.add_argument('--fs_sample_size', default=60000, type=int) 
    parser.add_argument('--save_model', action='store_true', help='Save model checkpoints')
    parser.add_argument('--big_data_cache_dir', type=str, default='./cache_data/ofa_dataset', help='Cache directory for zero-shot data and checkpoints')
    parser.add_argument('--checkpoint_interval', default=10, type=int) 

    parser.add_argument("--use_onecycle", action="store_true", help="Use onecycle scheduler")
    parser.add_argument('--max-lr', default=0.05, type=float)
    parser.add_argument('--div-factor', default=25, type=float)
    parser.add_argument('--final-div', default=10000, type=float)
    parser.add_argument('--pct-start', default=0.3, type=float)
    
    parser.add_argument('--prune_ratio', default=0.5, type=float, help='prune ratio') 
    parser.add_argument('--delta', default=1.0, type=float) 

    parser.add_argument('--hard_pruning_mode', type=str, default="random", help='Hard pruning mode')
    parser.add_argument('--hard_pruning_ratio', type=float, default=0, help='Hard pruning ratio')
    parser.add_argument('--hard_pruning_epochs', type=int, default=25, help='Hard pruning epochs')

    
    parser.add_argument('--hard_pruning_joint', action='store_true', help='Hard pruning joint')
    parser.add_argument('--hard_pruning_reverse', action='store_true', help='Hard pruning reverse')

    params = parser.parse_args()
    configs = []
    configs.append(
        load_yaml(
            os.path.join(
                os.path.dirname(__file__), "configs", "default_config.yaml"
            )
        )
    )

    if params.override is not None:
        override_config = load_yaml(params.override)
        configs.append(override_config)
    

    mod_params = combine_dict(*configs)
    mod_params = merge_mod(mod_params, params.opts)

    
    new_params = {
        k: v for k, v in vars(params).items() 
        if k not in mod_params
    }
    
    mod_params.update(new_params)
    setup_exp(mod_params)   

    params = SimpleNamespace(**mod_params)
    set_random_seed(params.seed)

    torch.set_float32_matmul_precision("high")
    params.log_project = "OFA_dcgfm"

    params.exp_name += f"dcgfm_{params.hard_pruning_mode}_{params.hard_pruning_epochs}_{params.hard_pruning_ratio}_{params.prune_ratio}"
    
    if params.control_gpu:
        params.gpus = [int(gpu_id) for gpu_id in params.gpus.split(",")]

    print(params)
    main(params)
