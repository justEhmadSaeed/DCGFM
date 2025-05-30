<h1 align="center"> Advancing Graph Foundation Models: A Data-Centric Perspective </h1>

This is an anonymous code repository for the KDD'25 submission of "Advancing Graph Foundation Models: A Data-Centric Perspective". The project is still under development.

**DCGFM** is a play-and-plug approach for data-centric GFM from the data pruning perspective. First, to mitigate redundancy and noise issues, a model-agnostic hard pruning module is employed to filter out less informative subgraphs. Second, to improve efficiency during training, a model-aware soft pruning module is utilized to filter out several pre-training subgraphs that contribute less to the current GFM in each epoch.

![DCGFM](images/framework.png)

## Setup Environment

```shell
conda create -n dcgfm python=3.11
conda activate dcgfm
pip install -r requirements.txt
```

We evaluate DCGFM on two representative GFM backbones, i.e., OFA and GraphCLIP.

## OFA with DCGFM


## GraphCLIP with DCGFM

### Data preparation

Please follow the GraphCLIP's repo to download the following pre-training datsaets and unzip the files and place them in the `summary` directory:

|Datasets  | Links |  
|--|--|
|OGBN-ArXiv|[Google Drive](https://drive.google.com/file/d/1AeAnnqPui05FuBX7JvWQMJA8kr2CIFYS/view?usp=sharing)|
| ArXiv\_2023| [Google Drive](https://drive.google.com/file/d/1t1icJvRtw9OBpc88uws_wIsKFoVHtM0D/view?usp=sharing)|
| Reddit|[Google Drive](https://drive.google.com/file/d/1c7gtoy918suLlUN5a8CYUGCEbzYAeSeX/view?usp=sharing) |
|OGBN-Products|[Google Drive](https://drive.google.com/file/d/1IAmU8mAJ-rVzFu1iOkvQes1RtS8-RU-M/view?usp=sharing)|


For target datasets, we only need to download processed data, unzip them and put them into `processed_data` directory:

|Datasets  | Links |  
|--|--|
|WikiCS|[Google Drive](https://drive.google.com/file/d/1vOo_Iql19Eccgr8t6H70AYIvxwu87846/view?usp=sharing)|
|Instagram|[Google Drive](https://drive.google.com/file/d/1c9ZkdHyDHKaInGnmXlLGjYIPeTY-njF7/view?usp=sharing)|
|Ele-Photo|[Google Drive](https://drive.google.com/file/d/1qFMixgszCODpo7e7syhucUjKYr75T8cx/view?usp=sharing)|
|Ele-Computers|[Google Drive](https://drive.google.com/file/d/1487we3C9AJryvAMCCH0W7YA0nXFQ1H8o/view?usp=sharing)|
|Books-History|[Google Drive](https://drive.google.com/file/d/1zAlK6BdQy0YmwPu9M5GXbImLrDQS4BON/view?usp=sharing)|

Please run `bash gen_target_subg.sh` to generate subgraphs for each target dataset.

### Pre-training from scratch with DCGFM

#### Step1: Model-agnostic Hard Pruning

```
cd GraphCLIP/

python hard_pruning.py --source_data ogbn-arxiv+arxiv_2023+pubmed+ogbn-products+reddit --threshold 30/50/70
```

#### Step2: Model-aware Soft Pruning

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --source_data ogbn-arxiv+arxiv_2023+pubmed+ogbn-products+reddit --batch_size 7200 --epochs 30
```

### Evaluation

```
CUDA_VISIBLE_DEVICES=0 python eval.py --target_data cora+citeseer+wikics+instagram+photo+computer+history --ckpt graphclip
```