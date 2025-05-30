import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar

from gp.lightning.metric import EvalKit
from gp.utils.utils import dict_res_summary, load_pretrained_state

import time


from lightning.pytorch.callbacks import Callback 
from gp.lightning.module_template import IBBaseTemplate
class InfoBatchCallback(Callback):
    def __init__(self):
        super().__init__()
        self.info_batch = None
    
    def on_train_start(self, trainer, pl_module):
        
        if not isinstance(pl_module, IBBaseTemplate):
            raise TypeError(
                f"pl_module 必须是 IBBaseTemplate 的实例，"
                f"但获得的是 {type(pl_module).__name__} 类型"
            )
        
        
        if not hasattr(trainer.datamodule, 'datasets'):
            raise AttributeError("datamodule 必须有 datasets 属性")
        
        self.info_batch = trainer.datamodule.datasets['train'].data
        
        pl_module.info_batch = self.info_batch


class TimerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
        
    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        
        trainer.logger.log_metrics({
            "epoch_time": epoch_time,
            "epoch": trainer.current_epoch
        })
        
        hours = epoch_time // 3600
        minutes = (epoch_time % 3600) // 60
        seconds = epoch_time % 60
        print(f"第 {trainer.current_epoch} 轮用时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

def lightning_fit(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    num_epochs,
    profiler=None,
    cktp_prefix="",
    load_best=True,
    prog_freq=20,
    test_rep=1,
    save_model=True,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
    reload_freq=0,
    val_interval=1,
    limit_val_batches=0,
    fs_sample_size=2000,
    checkpoint_interval=5,
    strategy=None,
):
    start_time = time.time()
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    if save_model:
        callbacks.append(
            ModelCheckpoint(
                save_top_k=-1, 
                save_last=True,
                every_n_epochs=checkpoint_interval,
                filename=cktp_prefix + "_{epoch}",
            )
        )
    
    callbacks.append(InfoBatchCallback())
    
    callbacks.append(TimerCallback())
    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        max_epochs=num_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_checkpointing=save_model,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
        reload_dataloaders_every_n_epochs=reload_freq,
        check_val_every_n_epoch=val_interval,
        limit_val_batches=limit_val_batches, 
    )
    trainer.fit(model, datamodule=data_module)
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")

    if logger:
        logger.log_metrics({
            "training_time_seconds": total_time,
            
        })

    if load_best:
        model_dir = trainer.checkpoint_callback.best_model_path
        deep_speed = False
        if strategy[:9] == "deepspeed":
            deep_speed = True
        state_dict = load_pretrained_state(model_dir, deep_speed)
        model.load_state_dict(state_dict)
    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))

    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    
    return [
        target_test_mean,
        target_test_std,
    ]


def lightning_test(
    logger,
    model,
    data_module,
    metrics: EvalKit,
    model_dir: str,
    strategy="auto",
    profiler=None,
    prog_freq=20,
    test_rep=1,
    prog_bar=True,
    accelerator="auto",
    detect_anomaly=False,
    deep_speed=True,
    fs_sample_size=2000,
):
    callbacks = []
    if prog_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))
    trainer = Trainer(
        accelerator=accelerator,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=prog_freq,
        profiler=profiler,
        enable_progress_bar=prog_bar,
        detect_anomaly=detect_anomaly,
    )
    state_dict = load_pretrained_state(model_dir, deep_speed)
    model.load_state_dict(state_dict)
    test_col = []
    for i in range(test_rep):
        test_col.append(
            trainer.test(model, datamodule=data_module, verbose=False)[0]
        )

    test_res = dict_res_summary(test_col)
    for met in test_res:
        test_mean = np.mean(test_res[met])
        test_std = np.std(test_res[met])
        print("{}:{:f}±{:f}".format(met, test_mean, test_std))
    
    target_test_mean = np.mean(test_res[metrics.test_metric])
    target_test_std = np.std(test_res[metrics.test_metric])
    
    return [
        target_test_mean,
        target_test_std,
    ]
