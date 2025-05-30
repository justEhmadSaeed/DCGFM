import torch
import pytorch_lightning as pl 
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import DataLoader
from test_tube import HyperOptArgumentParser

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class DeepGLAD(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, weight_decay=5e-4, **kwargs):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay 
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, data):
        
        raise NotImplementedError 

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), 
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)
        
    def test_step(self, batch, batch_idx): 
        self.test_step_outputs.append((self(batch)))
        
        return self(batch)

class GIN(nn.Module):
    """
    Note: batch normalization can prevent divergence maybe, take care of this later. 
    """
    def __init__(self,  nfeat, nhid, nlayer, dropout=0, act=ReLU(), bias=False, **kwargs):
        super(GIN, self).__init__()
        self.norm = BatchNorm1d
        self.nlayer = nlayer
        self.act = act
        self.transform = Sequential(Linear(nfeat, nhid), self.norm(nhid))
        self.pooling = global_mean_pool
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.nns = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(nlayer):
            self.nns.append(Sequential(Linear(nhid, nhid, bias=bias), 
                                       act, Linear(nhid, nhid, bias=bias)))
            self.convs.append(GINConv(self.nns[-1]))
            self.bns.append(self.norm(nhid))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.transform(x) 
        
        embed = self.pooling(x, batch)
        std = torch.sqrt(self.pooling((x - embed[batch])**2, batch))
        graph_embeds = [embed]
        graph_stds = [std]
        
        for i in range(self.nlayer):
            x = self.dropout(x)
            x = self.convs[i](x, edge_index)
            x = self.act(x)
            x = self.bns[i](x)
            embed = self.pooling(x, batch) 
            std = torch.sqrt(self.pooling((x - embed[batch])**2, batch))
            graph_embeds.append(embed)
            graph_stds.append(std)

        graph_embeds = torch.stack(graph_embeds)
        graph_stds = torch.stack(graph_stds)

        return graph_embeds, graph_stds
    
class GIN_Hard_Prune(DeepGLAD):
    def __init__(self, nfeat,
                 nhid=128, 
                 nlayer=3,
                 dropout=0, 
                 learning_rate=0.001,
                 weight_decay=0,
                 **kwargs):
        model = GIN(nfeat, nhid, nlayer=nlayer, dropout=dropout)
        super().__init__(model, learning_rate, weight_decay)
        self.save_hyperparameters() 
        self.radius = 0
        self.nu = 1
        self.eps = 0.01
        self.mode = 'sum' 
        assert self.mode in ['concat', 'sum']
        self.register_buffer('center', torch.zeros(nhid if self.mode=='sum' else (nlayer+1)*nhid ))
        self.register_buffer('all_layer_centers', torch.zeros(nlayer+1, nhid))
        
    def get_hiddens(self, data):
        embs, stds = self.model(data)
        return embs

    def forward(self, data):
        embs, stds = self.model(data)
        if self.mode == 'concat':
            embs = torch.cat([emb for emb in embs], dim=-1) 
        else:
            
            embs = embs.sum(dim=0) 

        dist = torch.sum((embs - self.center) ** 2, dim=1)
        scores = dist - self.radius ** 2
        return scores

    def training_step(self, batch, batch_idx):
        if self.current_epoch == 0:
            
            embs, stds = self.model(batch) 
            loss = torch.zeros(1, requires_grad=True, device=self.device) 
            self.training_step_outputs.append({'loss':loss, 'emb':embs.detach()})
            return {'loss':loss, 'emb':embs.detach()}
        else:
            assert self.center != None
            scores = self(batch)
            loss = self.radius ** 2 + (1 / self.nu) * torch.mean(F.relu(scores))
            self.training_step_outputs.append(loss)
            self.log('training_loss', loss)
            return loss

    def on_train_epoch_end(self):
        if self.current_epoch == 0:
            
            embs = torch.cat([d['emb'] for d in self.training_step_outputs], dim=1)
            self.all_layer_centers = embs.mean(dim=1)
            if self.mode == 'concat':
                self.center = torch.cat([x for x in self.all_layer_centers], dim=-1)
            else:
                self.center = torch.sum(self.all_layer_centers, 0)
        else:
            
            losses = [item for item in self.training_step_outputs if isinstance(item, torch.Tensor)]
            if losses:  
                avg_loss = torch.stack(losses).mean()
                
                self.log('epoch_avg_loss', avg_loss, prog_bar=True)
                print(f"Epoch {self.current_epoch} 平均loss: {avg_loss.item():.6f}")
        
        self.training_step_outputs = []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser], add_help=False)
        
        parser.add_argument('--nhid', type=int, default=32)
        parser.add_argument('--nlayer', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0)
        
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        return parser

class SimpleDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.')  
        self.data, self.slices = self.collate(data_list)
    
    @property
    def raw_file_names(self): return []
    
    @property
    def processed_file_names(self): return []
    def download(self): pass
    def process(self): pass

def hard_prune_api(data_list, batch_size=32, weight_decay=5e-4, nlayer=5, max_epochs=25, devices=[0]):
    dataset = SimpleDataset(data_list)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = GIN_Hard_Prune(dataset[0].x.shape[1], weight_decay=weight_decay, nlayer=nlayer)
    trainer = pl.Trainer(accelerator='gpu', devices=devices, max_epochs=max_epochs, logger=True)
    trainer.fit(model=model, train_dataloaders=train_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader)
    
    if not model.test_step_outputs:
        raise ValueError("No test outputs available")
        
    local_scores = torch.cat([out for out in model.test_step_outputs])
    if trainer.world_size > 1:
        gathered_scores = trainer.strategy.all_gather(local_scores)
        if trainer.is_global_zero:  
            anomaly_scores = torch.cat([scores for scores in gathered_scores])
            return anomaly_scores.cpu().detach().numpy()
    else:
        return local_scores.cpu().detach().numpy()

def create_valid_edge_index(num_nodes, num_edges):
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index

if __name__ == '__main__':
    tmp_data_list = []
    tmp_dataset_size = 100
    for i in range(tmp_dataset_size):
        num_nodes = 10
        num_edges = 10
        
        edge_index = create_valid_edge_index(num_nodes, num_edges)
        actual_edges = edge_index.size(1)
        
        tmp_data_list.append(Data(
            x=torch.randn(num_nodes, 10),           
            edge_index=edge_index,                  
            y=torch.randint(0, 2, (1,)),           
        ))
    print(tmp_data_list)
    anomaly_scores = hard_prune_api(tmp_data_list, batch_size=32, weight_decay=5e-4, nlayer=5, devices=[0])
    print(anomaly_scores)