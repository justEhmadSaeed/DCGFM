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

def get_useful_indices(data_list_no_prompt, hard_pruning_mode, hard_pruning_ratio, hard_pruning_reverse, hard_pruning_epochs):
    anomany_scores = globals()[hard_pruning_mode](data_list = data_list_no_prompt, max_epochs = hard_pruning_epochs)
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
            if params.hard_pruning_joint == True: 
                useful_indices_all = get_useful_indices(data_list_no_link, params.hard_pruning_mode, params.hard_pruning_ratio, params.hard_pruning_reverse, params.hard_pruning_epochs)
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
                    effective_indices[i] = get_useful_indices(data_list_no_prompt_ind[i], params.hard_pruning_mode, params.hard_pruning_ratio, params.hard_pruning_reverse, params.hard_pruning_epochs)
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
