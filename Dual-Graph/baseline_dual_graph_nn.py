import argparse
import torch
import torch.nn.functional as F

import sklearn.metrics, sklearn.preprocessing
import argparse, dgl, numpy, ast, torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd
import dgl.nn.pytorch as dglnn

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset
from evaluator import Evaluator
from logger import Logger

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x




class TwoInputsGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, device, use_sage):
        super(TwoInputsGCN, self).__init__()

        if use_sage:
            self.model = SAGE(in_channels, hidden_channels, 128,
                     num_layers, dropout).to(device)
        else:
            self.model = GCN(in_channels, hidden_channels, 128,
                    num_layers, dropout).to(device)

        self.linear = nn.Linear(128, out_channels).to(device)

    

    def reset_parameters(self):
        self.model.reset_parameters()
        self.linear.reset_parameters()



    def forward(self, input1, in_feat1):

        h1 = self.model(input1, in_feat1)
        out = self.linear(h1)  # add layer
        return out


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()
    return loss.item(), criterion

@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()
    y_pred = model(data.x, data.adj_t)
    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })
    return train_rocauc["rocauc"], valid_rocauc["rocauc"],test_rocauc["rocauc"],train_rocauc["acc"], valid_rocauc["acc"],test_rocauc["acc"], y_pred[split_idx['test']], data.y[split_idx['test']]


def save_checkpoint(epoch,model,optimizer,loss_f,path):
    torch.save({'epoch':epoch, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'loss': loss_f}, path)

def save_output(predictions, x, path_output):
    file_predictions = open(path_output, 'w')
    file_predictions.write('Predicted_output\tExpected_Output\n')
    for i in range(len(x)):
        file_predictions.write(str(predictions[i]) + '\t' + str(x[i]) + '\n')
    file_predictions.close()




def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]
    data.x = data.adj_t.mean(dim=1)
    print(data.x)
    print(len(data.x))
    print(len(data.x[0]))

    data.adj_t.set_value_(None)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    model = TwoInputsGCN(data.num_features, args.hidden_channels, 112, args.num_layers,
                 args.dropout, device, args.use_sage)
    if args.use_sage:
        pass
    else:
        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    max_result = 0

    data = data.to(device)
    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss, criterion = train(model, data, train_idx, optimizer)
            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)
                trocauc, vrocauc,  terocauc, tracc, vacc, teacc, ypred, ytrue = result
                if vrocauc > max_result and epoch > 700:
                    max_result = vrocauc
                    save_output(ypred, ytrue, "./test_predictions_baseline.txt")
                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc,  test_rocauc, train_acc, valid_acc, test_acc, test_y_pred, test_y_true = result
                    if len(result) >  3:
                        print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}% '
                          f'Acc Train: {100 * train_acc:.2f}% '
                          f'Acc Valid: {100 * valid_acc:.2f}% '
                          f'Acc Test: {100 * test_acc:.2f}%')

                    save_checkpoint(epoch,model,optimizer,criterion, "./model_dgnn_baseline.pt")

        logger.print_statistics(run)
    logger.print_statistics()

if __name__ == "__main__":
    main()
