import argparse

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from ogb.nodeproppred import PygNodePropPredDataset
from evaluate import Evaluator

from logger import Logger


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            print(_)
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def train(model, x, y_true, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer.zero_grad()
    out = model(x)[train_idx]
    loss = criterion(out, y_true[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator):
    model.eval()

    y_pred = model(x)
    train_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })
    valid_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })
    test_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })

    return train_rocauc["rocauc"], valid_rocauc["rocauc"],test_rocauc["rocauc"],train_rocauc["acc"], valid_rocauc["acc"],test_rocauc["acc"]

def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (MLP)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_node_embedding', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--agr', action='store_true')
    parser.add_argument('--fullgo', action='store_true')
    parser.add_argument('--gen', action='store_true')
    parser.add_argument('--col8', action='store_true')
    parser.add_argument('--col9', action='store_true')
    parser.add_argument('--labels', type=int, default=112)

    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins')
    split_idx = dataset.get_idx_split()
    data = dataset[0]

    data_edge1 = data.edge_attr[:79122504,:8]
    data_edge2 = data.edge_attr[79122504:,:8]
    data_edge3 = data.edge_attr[79122504:,8:]
    data_edge4 = data.edge_attr[79122504:,:1]
    
    edge_ind1 = data.edge_index[:,:79122504]
    edge_ind2 = data.edge_index[:,79122504:]
    data_orignodes = 132534
    if args.gen:
        num_node_gen = 132776
        if args.col8:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_gen, reduce='mean').to('cpu')
            x2 = x2[data_orignodes:]
            x = torch.cat((x1,x2),0)

        if args.col9:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_gen, reduce='mean').to('cpu')
            x3 = scatter(data_edge3, edge_ind2[0], dim=0,
                       dim_size=num_node_gen, reduce='mean').to('cpu')
            x4 = scatter(data_edge4, edge_ind2[0], dim=0,
                       dim_size=num_node_gen, reduce='mean').to('cpu')
            x2 = x2[data_orignodes:]
            x = torch.cat((x1,x2),0)
            x3 = x3[:data_orignodes]
            x4 = x4[data_orignodes:]
            x4 = torch.cat((x3,x4),0)
            x = torch.cat((x,x4),-1)

    if args.fullgo:
        num_node_go = 176798
        if args.col8:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_go, reduce='mean').to('cpu')
            x2 = x2[data_orignodes:]
            x = torch.cat((x1,x2),0)

        if args.col9:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_go, reduce='mean').to('cpu')
            x3 = scatter(data_edge3, edge_ind2[0], dim=0,
                       dim_size=num_node_go, reduce='mean').to('cpu')
            x4 = scatter(data_edge4, edge_ind2[0], dim=0,
                       dim_size=num_node_go, reduce='mean').to('cpu')
            x2 = x2[data_orignodes:]
            x = torch.cat((x1,x2),0)
            x3 = x3[:data_orignodes]
            x4 = x4[data_orignodes:]
            x4 = torch.cat((x3,x4),0)
            x = torch.cat((x,x4),-1)

    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location='cpu')
        x = torch.cat([x, embedding], dim=-1)
        
    if args.use_save_embedding:      
        torch.save(x, "embedding_forgnn.pt")

    print(len(x))
    print(len(x[0]))
    print(x)

    x = x.to(device) 
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)
    model = MLP(x.size(-1), args.hidden_channels, args.labels, args.num_layers,
                args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-proteins', nlabels=args.labels)
    logger = Logger(args.runs, args)
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, x, y_true, train_idx, optimizer)
            if epoch % args.eval_steps == 0:
                result = test(model, x, y_true, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc,  test_rocauc, train_acc, valid_acc, test_acc = result

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

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
