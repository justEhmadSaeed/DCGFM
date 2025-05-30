import torch
from torch_geometric import seed_everything
from utils.process import parse_source_data
from kernal import kernel_api
from torch_geometric.data import DataLoader
import argparse
from hard_prune_module import hard_prune_api
import numpy as np

# python hard_pruning.py --source_data ogbn-arxiv+arxiv_2023+pubmed+ogbn-products+reddit --threshold 30

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data', type=str, required=True,
                      help='Source data names separated by +')
    parser.add_argument('--threshold', type=int, default=70,
                      help='Percentile threshold for filtering (50-70 recommended)')
    parser.add_argument('--save_path', type=str, default="hard_pruning_datasets/_graphs_{method}_t{threshold}.pt",
                      help='Path to save the filtered graphs')
    parser.add_argument('--seed', type=int, default=88,
                      help='Random seed')
    return parser.parse_args()

def save_graphs(args):
    # collect source data
    all_source_graph = []
    source_name_list = args.source_data.split("+")
    for source_name in source_name_list:
        source_data = torch.load(f"processed_data/{source_name}.pt")
        source_graph = parse_source_data(source_name, source_data)
        all_source_graph.extend(source_graph)
    
    original_count = len(all_source_graph)
    print(f"Loaded {original_count} graphs in total")

    anomaly_scores = hard_prune_api(
        all_source_graph,
        batch_size=512,
        weight_decay=5e-4,
        nlayer=5,
        max_epochs=25,
        devices=0
    )
        
    # Keep graphs with lowest anomaly scores based on threshold
    threshold = np.percentile(anomaly_scores, args.threshold)
    filtered_graphs = [g for i, g in enumerate(all_source_graph) 
                        if anomaly_scores[i] <= threshold]
    
    filtered_count = len(filtered_graphs)
    print(f"Filtered to {filtered_count} non-anomalous graphs using {args.method} method")
    
    # Save filtered graphs with method and threshold in filename
    save_path = args.save_path.format(method=args.method, threshold=args.threshold)
    torch.save(filtered_graphs, save_path)
    print(f"Saved non-anomalous graphs to {save_path}")
    
    return filtered_graphs, original_count, filtered_count

def random_select_graphs(args):
    # collect source data
    all_source_graph = []
    source_name_list = args.source_data.split("+")
    for source_name in source_name_list:
        source_data = torch.load(f"processed_data/{source_name}.pt")
        source_graph = parse_source_data(source_name, source_data)
        all_source_graph.extend(source_graph)
    
    original_count = len(all_source_graph)
    print(f"Loaded {original_count} graphs in total")

    # Randomly select 30% of graphs
    num_select = int(original_count * 0.3)
    selected_indices = np.random.choice(original_count, num_select, replace=False)
    filtered_graphs = [all_source_graph[i] for i in selected_indices]
    
    filtered_count = len(filtered_graphs)
    print(f"Randomly selected {filtered_count} graphs (30% of original)")
    
    # Save filtered graphs
    save_path = args.save_path.format(method="random", threshold=30)
    torch.save(filtered_graphs, save_path)
    print(f"Saved randomly selected graphs to {save_path}")
    
    return filtered_graphs, original_count, filtered_count

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    
    filtered_graphs, orig_count, filtered_count = save_graphs(args)
    # filtered_graphs, orig_count, filtered_count = random_select_graphs(args)
