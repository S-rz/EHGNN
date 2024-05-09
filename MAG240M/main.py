import argparse
import torch
import time
import warnings
import numpy as np
from utils import random_walk_sim, accuracy, get_model_need, load_240m, get_need_nodes, load_features
from models import EHGNN

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../data/', help='path of dataset')
    parser.add_argument('--feature_path', type=str, default='../data/mag240m_kddcup2021/processed/paper/', help='path of paper features')
    parser.add_argument('--other_feature_path', type=str, default='../data/mag240m_kddcup2021/', help='path of other features')
    parser.add_argument('--is_normalize', action='store_true', help='Is row normalize for features')
    parser.add_argument('--wo_l2', action='store_true', help='without l2 normalization of output')
    parser.add_argument('--wo_mweight', action='store_true', help='without meta-path weight')
    parser.add_argument('--wo_tweight', action='store_true', help='without node type weight')
    parser.add_argument('--r_neighbor', action='store_true', help='random choice neighborhoods')
    parser.add_argument('--K_a', type=int, default=10, help='Top K of similarity, number of author per node')
    parser.add_argument('--K_i', type=int, default=5, help='Top K of similarity, number of institution per node')
    parser.add_argument('--K_p', type=int, default=20, help='Top K of similarity, number of paper per node')
    parser.add_argument('--walk_num', type=int, default=40, help='number of meta-path random walk per node')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dimension of mlp layer')
    parser.add_argument('--n_layers', type=int, default=4, help='number of mlp layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha of ppr')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--val_epochs', type=int, default=5, help='Number of epochs to valid')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=3000, help='batch size')
    parser.add_argument('--gpu', type=int, default=0, help='number of device')
    return parser.parse_args()

metapaths = []
## 'cites', 'writes', 'affiliated_with'
metapaths.append(['writes_r', 'writes', 'writes_r', 'writes'])
metapaths.append(['cites', 'cites', 'cites', 'cites'])
metapaths.append(['cites_r', 'cites', 'cites_r', 'cites'])
metapaths.append(['writes_r', 'affiliated_with', 'affiliated_with_r', 'writes'])

if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    print('My similarity.')
    print(metapaths)
    start = time.perf_counter()
    g, labels, idx_train, idx_test = load_240m(args.path)
    end = time.perf_counter()
    print('Done Load Graph, Running time: {:.4f} Seconds'.format(end - start))

    top_k = [args.K_a, args.K_i, args.K_p]

    start = time.perf_counter()
    train_matrixs = []
    test_matrixs = []
    t_typess = []
    for metapath in metapaths:
        train_matrix, t_types, s_type = random_walk_sim(idx_train, g, metapath, args.walk_num, top_k, args.r_neighbor)
        test_matrix, _, _ = random_walk_sim(idx_test, g, metapath, args.walk_num, top_k, args.r_neighbor)

        train_matrixs.append(train_matrix)
        test_matrixs.append(test_matrix)
        t_typess.append(t_types)
    end = time.perf_counter()
    print('Done sim, Running time: {:.4f} Seconds'.format(end - start))

    ## type : set
    train_related_nodes = get_need_nodes(train_matrixs, 3, t_typess)
    test_related_nodes = get_need_nodes(test_matrixs, 3, t_typess)
    all_related_nodes = []
    related_num = 0
    for i in range(3):
        all_related_nodes.append(list(train_related_nodes[i].union(test_related_nodes[i])))
        related_num += len(all_related_nodes[i])
    print('Related Nodes Num : ', related_num)

    print('Load Feature')
    start = time.perf_counter()
    features, features_map = load_features(all_related_nodes, args.feature_path, args.other_feature_path, args.is_normalize)
    end = time.perf_counter()
    print('Done Feature, Running time: {:.4f} Seconds'.format(end - start))

    out_dim = labels.max().item() + 1
    model = EHGNN(in_feat=features[0].shape[1],
                  hidden=args.hidden,
                  out_feat=out_dim,
                  n_layer=args.n_layers,
                  alpha=args.alpha,
                  n_metapath=len(metapaths),
                  n_types=len(g.ntypes),
                  wo_l2=args.wo_l2,
                  wo_mweight=args.wo_mweight,
                  wo_tweight=args.wo_tweight,
                  dropout=args.dropout,
                  ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fcn = torch.nn.NLLLoss()

    print('Begin Train.')
    for run in range(args.epochs):
        start = time.perf_counter()
        dataloader = torch.utils.data.DataLoader(idx_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
        model.train()
        loss_avg = []
        ma_avg = []
        mi_avg =[]
        for batch in dataloader:
            optimizer.zero_grad()
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths), train_matrixs, t_typess, batch)
            map_batch = features_map[s_type][batch]
            batch_out = model(features, features[s_type][map_batch], features_map, s_idxs, t_idxs, weightss, t_typess, batch.shape[0], device)
            y_true = labels[batch].to(device)
            loss = loss_fcn(batch_out, y_true.squeeze(dim=1))
            loss_avg.append(loss.item())
            macro_f1, micro_f1 = accuracy(batch_out, y_true)
            ma_avg.append(macro_f1)
            mi_avg.append(micro_f1)
            loss.backward()
            optimizer.step()

        end = time.perf_counter()
        print('Epoch : {}, loss : {:.4f}, macro f1 : {:.4f}, micro f1 : {:.4f}, Running time: {:.4f} Seconds'.
              format(run, np.array(loss_avg).mean(), np.array(ma_avg).mean(), np.array(mi_avg).mean(), end - start))

        if run % args.val_epochs == 0 and run != 0:
            with torch.no_grad():
                model.eval()

                ## Test
                start = time.perf_counter()
                test_loader = torch.utils.data.DataLoader(idx_test, batch_size=args.batch_size, shuffle=False, drop_last=False)
                test_out = torch.FloatTensor([])
                for batch_test in test_loader:
                    s_idxs, t_idxs, weightss = get_model_need(len(metapaths), test_matrixs, t_typess, batch_test)
                    map_batch = features_map[s_type][batch_test]
                    result = model(features, features[s_type][map_batch], features_map, s_idxs, t_idxs, weightss, t_typess, batch_test.shape[0], device).to('cpu')
                    test_out = torch.cat((test_out, result), dim=0)

                y_true = labels[idx_test]
                macro_f1, micro_f1 = accuracy(test_out, y_true)
                end = time.perf_counter()
                print('Test macro f1 : {:.4f}, micro f1 : {:.4f}, Time : {:.4f}'.format(macro_f1, micro_f1, end-start))