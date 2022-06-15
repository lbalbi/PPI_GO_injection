import argparse

import torch
import torch.nn.functional as F
from torch_scatter import scatter

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        ### (x.size(-1), args.hidden_channels, 112, args.num_layers,args.dropout)
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
    out = model(x)[train_idx] # tensor w/ values between 0 and 1  with 86619 rows and 112 cols
    loss = criterion(out, y_true[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, x, y_true, split_idx, evaluator): # y_true is a 132534 row and 112 col tensor with 1s and 0s
    model.eval()

    y_pred = model(x) # length is 132534
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

        # try:
        #     results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits[f'hits@{K}'],test_hits["roc_auc"], test_hits["recall"],test_hits["precision"])
        # except:
        #     results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits[f'hits@{K}'])
    return train_rocauc["rocauc"], valid_rocauc["rocauc"],test_rocauc["rocauc"],train_rocauc["acc"], valid_rocauc["acc"],test_rocauc["acc"]
    #return train_rocauc, valid_rocauc["rocauc"],valid_rocauc["acc"],valid_rocauc["recall"],valid_rocauc["precision"], test_rocauc["rocauc"], test_rocauc["acc"], test_rocauc["recall"], test_rocauc["precision"]


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
    split_idx = dataset.get_idx_split() # split_idx é um dic onde "treino" possui os primeiros 86619 node idx, "valid" possui 
    # os 88619 a 107855 seguintes; "teste" possui node idx 107856 a 132533.
    data = dataset[0] # aspeto: Data(edge_index=[2,79122504],edge_attr=[79122504,8],node_species=[132534,1],y=[132534,112])
    #print(data)

    #data_edge = data.edge_attr[:,:8]
    data_edge1 = data.edge_attr[:79122504,:8]
    data_edge2 = data.edge_attr[79122504:,:8]
    data_edge3 = data.edge_attr[79122504:,8:]
    data_edge4 = data.edge_attr[79122504:,:1]
    #data_edge4 = data.edge_attr[79122505:81577237:2,8:] ## Annots original
    #data_edge5 = data.edge_attr[81720267::2,8:] ## Annots original
    #data_edge4 = data.edge_attr[79122505:81391353:2,8:] ## Annots random
    #data_edge5 = data.edge_attr[81534383::2,8:] ## Annots random
    #data_edge4 = data.edge_attr[79122505:81338635:2,8:] ## Annots random 2 sTestPG
    #data_edge5 = data.edge_attr[81481665::2,8:] ## Annots random 2 sTestPG
    #data_edge5 = torch.cat((data_edge4,data_edge5),0)
    ##data_edge5 = data.edge_attr[81338634:81481664,8:] ## Annots random 2 sTestPG
    ##data_edge5 = data.edge_attr[81391352:81534382,8:] ## Annots random
    #data_edge5 = data.edge_attr[81577236:81720266:2,8:] ## Annots original

    #edge_ind = data.edge_index[:,:]
    edge_ind1 = data.edge_index[:,:79122504]
    edge_ind2 = data.edge_index[:,79122504:]
    ##edge_ind3 = data.edge_index[:,79122504:]
    #edge_ind4 = data.edge_index[:,79122505:81577237:2] ## Annots original
    #edge_ind5 = data.edge_index[:,81720267::2] ## Annots original
    #edge_ind4 = data.edge_index[:,79122505:81391353:2] ## Annots random
    #edge_ind5 = data.edge_index[:,81534383::2] ## Annots random
    #edge_ind4 = data.edge_index[:,79122505:81338635:2] ## Annots random 2 sTestPG
    #edge_ind5 = data.edge_index[:,81481665::2] ## Annots random 2 sTestPG
    #edge_ind5 = torch.cat((edge_ind4,edge_ind5),-1)
    #print(edge_ind5)
    #edge_ind5 = data.edge_index[:,81338634:81481664]
    #edge_ind5 = data.edge_index[:,81391352:81534382] ## Annots random
    #edge_ind5 = data.edge_index[:,81577236:81720266:2] ## Annots original

    data_orignodes = 132534

    if args.gen:
        num_node_gen = 132776
        if args.col8:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')  ## PPIS
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_gen, reduce='mean').to('cpu')  ## ZEROS
            x2 = x2[data_orignodes:]
            print(len(x1))
            print(len(x1[0]))
            print(len(x2))
            print(len(x2[0]))
            x = torch.cat((x1,x2),0)

        if args.col9:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')  ## PPIS
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_gen, reduce='mean').to('cpu')  ## ZEROS
            x3 = scatter(data_edge3, edge_ind2[0], dim=0,
                       dim_size=num_node_gen, reduce='mean').to('cpu')  ## MEDIA P PROTS
            x4 = scatter(data_edge4, edge_ind2[0], dim=0,
                       dim_size=num_node_gen, reduce='mean').to('cpu')  ## MEDIA P GOs
            x2 = x2[data_orignodes:]
            print(len(x1))
            print(len(x1[0]))
            print(len(x2))
            print(len(x2[0]))
            x = torch.cat((x1,x2),0)
            print(len(x))
            print(len(x[0]))
            print(x)
            x3 = x3[:data_orignodes]
            print(x3)
            print(len(x3))
            print(len(x3[0]))
            x4 = x4[data_orignodes:]
            #print(x4)
            print(len(x4))
            print(len(x4[0]))
            x4 = torch.cat((x3,x4),0)
            x = torch.cat((x,x4),-1)
        print(len(x))
        print(len(x[0]))
        print(x)


    if args.fullgo:
        num_node_go = 176798
        if args.col8:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')  ## PPIS
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_go, reduce='mean').to('cpu')  ## ZEROS
            x2 = x2[data_orignodes:]
            print(len(x1))
            print(len(x1[0]))
            print(len(x2))
            print(len(x2[0]))
            x = torch.cat((x1,x2),0)

        if args.col9:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')  ## PPIS
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_go, reduce='mean').to('cpu')  ## ZEROS
            x3 = scatter(data_edge3, edge_ind2[0], dim=0,
                       dim_size=num_node_go, reduce='mean').to('cpu')  ## MEDIA P PROTS
            x4 = scatter(data_edge4, edge_ind2[0], dim=0,
                       dim_size=num_node_go, reduce='mean').to('cpu')  ## MEDIA P GOs
            x2 = x2[data_orignodes:]
            print(len(x1))
            print(len(x1[0]))
            print(len(x2))
            print(len(x2[0]))
            x = torch.cat((x1,x2),0)
            print(len(x))
            print(len(x[0]))
            print(x)
            x3 = x3[:data_orignodes]
            print(x3)
            print(len(x3))
            print(len(x3[0]))
            x4 = x4[data_orignodes:]
            #print(x4)
            print(len(x4))
            print(len(x4[0]))
            x4 = torch.cat((x3,x4),0)
            x = torch.cat((x,x4),-1)
        print(len(x))
        print(len(x[0]))
        print(x)
            

    if args.agr:
        num_node_agr = 132651
        if args.col8:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')  ## PPIS
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_agr, reduce='mean').to('cpu')  ## ZEROS
            x2 = x2[data_orignodes:]
            print(len(x1))
            print(len(x1[0]))
            print(len(x2))
            print(len(x2[0]))
            x = torch.cat((x1,x2),0)

        if args.col9:
            x1 = scatter(data_edge1, edge_ind1[0], dim=0,
                       dim_size=data_orignodes, reduce='mean').to('cpu')  ## PPIS
            x2 = scatter(data_edge2, edge_ind2[0], dim=0,
                       dim_size=num_node_agr, reduce='mean').to('cpu')  ## ZEROS
            x3 = scatter(data_edge3, edge_ind2[0], dim=0,
                       dim_size=num_node_agr, reduce='mean').to('cpu')  ## MEDIA P PROTS
            x4 = scatter(data_edge4, edge_ind2[0], dim=0,
                       dim_size=num_node_agr, reduce='mean').to('cpu')  ## MEDIA P GOs
            x2 = x2[data_orignodes:]
            print(len(x1))
            print(len(x1[0]))
            print(len(x2))
            print(len(x2[0]))
            x = torch.cat((x1,x2),0)
            print(len(x))
            print(len(x[0]))
            print(x)
            x3 = x3[:data_orignodes]
            print(x3)
            print(len(x3))
            print(len(x3[0]))
            x4 = x4[data_orignodes:]
            #print(x4)
            print(len(x4))
            print(len(x4[0]))
            x4 = torch.cat((x3,x4),0)
            x = torch.cat((x,x4),-1)
        print(len(x))
        print(len(x[0]))
        print(x)
        


    #x = scatter(data_edge, edge_ind[0], dim=0,
    #            dim_size=data.num_nodes, reduce='mean').to('cpu')
    ################################
    #x1 = scatter(data_edge1, edge_ind1[0], dim=0,
    #           dim_size=data_orignodes, reduce='mean').to('cpu')  ## PPIS
    #x2 = scatter(data_edge2, edge_ind2[0], dim=0,
    #           dim_size=num_node_agr, reduce='mean').to('cpu')  ## ZEROS
    #x3 = scatter(data_edge3, edge_ind2[0], dim=0,
    #            dim_size=num_node_gen, reduce='mean').to('cpu')  ## MEDIA P PROTS
    #x4 = scatter(data_edge4, edge_ind2[0], dim=0,
    #            dim_size=num_node_gen, reduce='mean').to('cpu')  ## MEDIA P GOs
    #x4 = scatter(data_edge5, edge_ind5[0], dim=0,
    #            dim_size=num_node2, reduce='mean').to('cpu')  ## MEDIA P GOs
    
    #x2 = x2[data_orignodes:]
    #print(x1)
    #print(len(x1))
    #print(len(x1[0]))
    #print("-----")
    #print(x2)
    #print(len(x2))
    #print(len(x2[0]))
    #x = torch.cat((x1,x2),0)
    #print(len(x))
    #print(len(x[0]))
    #print(x)
    #x3 = x3[:132534]
    #print(x3)
    #print(len(x3))
    #print(len(x3[0]))
    #x4 = x4[132534:]
    #print(x4)
    #print(len(x4))
    #print(len(x4[0]))
    #x4 = torch.cat((x3,x4),0)
    ###x = torch.cat((x,x3),-1)
    #print(x4)
    #print(len(x4))
    #print(len(x4[0]))
    #x = torch.cat((x,x4),-1)
    #x = torch.cat((x,x2),-1)
    print(len(x))
    print(len(x[0]))
    print(x)

    torch.save(x, "8col_orig_with0s_forgnn.pt")


    #embeddingPPI = torch.load('embedding_PPI.pt')
    #emb = torch.zeros([132776-132534,128])     # go - 176798 ; gen - 132776 ; agr - 132651
    #embeddingPPI = torch.cat([embeddingPPI,emb], axis=0)
    #x = torch.cat([embeddingPPI, torch.load('embedding_GO.pt')], dim=-1)
    #print(x)
    #print(len(x))
    #print(len(x[0]))


    if args.use_node_embedding:
        embedding = torch.load('embedding.pt', map_location='cpu')
        x = torch.cat([x, embedding], dim=-1) #  x é um tensor exatamente igual em estrutura e conteúdo


    #embedding1 = torch.load('embedding_onlyPPIs.pt', map_location='cpu')
    #emb = torch.zeros([176800-132534,128])
    #embedding1 = torch.cat([embedding1,emb], axis=0)
    #embedding2 = torch.load('embedding_trainonlyGO.pt', map_location='cpu')
    #x = torch.cat([embedding1,embedding2], axis=-1)
    #x = torch.load('embedding_train.pt', map_location='cpu')
    print(len(x))
    print(len(x[0]))
    print(x)

    x = x.to(device) 
    y_true = data.y.to(device)  # y_true é um tensor de 132534 rows e 112 cols com 1s e 0s
    train_idx = split_idx['train'].to(device)  # train_idx é um tensor de um 1dim com os 1os 86619 node idx
    # com os elementos ordenados desde o 0
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
                    #train_rocauc, valid_rocauc, test_rocauc = result

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
