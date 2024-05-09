import argparse
import torch
import time
from utils import random_walk_sim, get_model_need, accuracy, neg_sample, load_Yelp
from models import EHGNN_yelp

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Yelp', help='dataset')
    parser.add_argument('--path', type=str, default='../data/', help='path of dataset')
    parser.add_argument('--is_normalize', action='store_true', help='Is row normalize for features')
    parser.add_argument('--wo_l2', action='store_true', help='without l2 normalization of output')
    parser.add_argument('--wo_mweight', action='store_true', help='without meta-path weight')
    parser.add_argument('--wo_tweight', action='store_true', help='without node type weight')
    parser.add_argument('--r_neighbor', action='store_true', help='random choice neighborhoods')
    parser.add_argument('--K', type=int, default=10, help='Top K of similarity, number of neighbors per node')
    parser.add_argument('--walk_num', type=int, default=100, help='number of meta-path random walk per node')
    parser.add_argument('--hidden', type=int, default=256, help='hidden dimension of mlp layer')
    parser.add_argument('--n_layers', type=int, default=2, help='number of mlp layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--eps', type=float, default=1e-5, help='eps of ppr')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha of ppr')
    parser.add_argument('--num_threads', type=int, default=40, help='number of threads for ppr random walk')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--val_epochs', type=int, default=5, help='Number of epochs to valid')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4000, help='batch size')
    parser.add_argument('--gpu', type=int, default=0, help='number of device')
    return parser.parse_args()

metapaths_b = []
metapaths_b.append(['locatedin', 'locatedin_r'])
metapaths_b.append(['rate', 'rate_r'])
metapaths_b.append(['describedwith', 'context', 'describedwith_r'])

metapaths_p = []
metapaths_p.append(['context_r', 'context'])
metapaths_p.append(['context', 'context_r', 'context'])
metapaths_p.append(['describedwith_r', 'describedwith'])

## Yelp
if __name__ == '__main__':
    args = parse_args()
    print(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    print(metapaths_b)
    print(metapaths_p)
    start = time.perf_counter()
    g, features, train_links, test_links, labels = load_Yelp(args.path, args.dataset, args.is_normalize)

    num_business = 7474
    num_phrase = 74943

    end = time.perf_counter()
    print('Done Load Data, Running time: {:.4f} Seconds'.format(end - start))

    start = time.perf_counter()
    sim_matrixs_b = []
    t_typess_b = []
    for metapath in metapaths_b:
        sim_matrix, t_types, s_type_b = random_walk_sim(torch.LongTensor([_ for _ in range(num_business)]), g, metapath, args.walk_num, args.K, args.r_neighbor)
        sim_matrixs_b.append(sim_matrix)
        t_typess_b.append(t_types)

    sim_matrixs_p = []
    t_typess_p = []
    for metapath in metapaths_p:
        sim_matrix, t_types, s_type_p = random_walk_sim(torch.LongTensor([_ for _ in range(num_phrase)]), g, metapath, args.walk_num, args.K, args.r_neighbor)
        sim_matrixs_p.append(sim_matrix)
        t_typess_p.append(t_types)
    end = time.perf_counter()
    print('Done my sim, Running time: {:.4f} Seconds'.format(end - start))

    model = EHGNN_yelp(in_feat=features[0].shape[1],
                  hidden=args.hidden,
                  out_feat=args.hidden,
                  n_layer=args.n_layers,
                  alpha=args.alpha,
                  n_metapath=len(metapaths_b) + len(metapaths_p),
                  n_types=len(g.ntypes) * 2,
                  wo_l2=args.wo_l2,
                  wo_mweight=args.wo_mweight,
                  wo_tweight=args.wo_tweight,
                  dropout=args.dropout,
                  ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fcn = torch.nn.BCELoss()

    print('Begin Train.')
    for run in range(args.epochs):
        loss_avg = []
        auc_avg = []
        precision_avg = []
        train_idx = torch.randint(0, train_links.shape[0], (num_phrase,))
        train_loader = torch.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
        step = 0
        for batch_train in train_loader:
            model.train()
            step = step + 1
            start = time.perf_counter()
            optimizer.zero_grad()
            pos_s = train_links[0][batch_train]
            pos_t = train_links[1][batch_train]
            neg_s, neg_t = neg_sample(pos_s, 0, num_phrase)

            s_idxs, t_idxs, weightss = get_model_need(len(metapaths_b), sim_matrixs_b, t_typess_b, pos_s)
            pos_s_out = model(features, features[s_type_b][pos_s], s_idxs, t_idxs, weightss, t_typess_b, 0, pos_s.shape[0], device)
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths_p), sim_matrixs_p, t_typess_p, pos_t)
            pos_t_out = model(features, features[s_type_p][pos_t], s_idxs, t_idxs, weightss, t_typess_p, 1, pos_t.shape[0], device)
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths_b), sim_matrixs_b, t_typess_b, neg_s)
            neg_s_out = model(features, features[s_type_b][neg_s], s_idxs, t_idxs, weightss, t_typess_b, 0, neg_s.shape[0], device)
            s_idxs, t_idxs, weightss = get_model_need(len(metapaths_p), sim_matrixs_p, t_typess_p, neg_t)
            neg_t_out = model(features, features[s_type_p][neg_t], s_idxs, t_idxs, weightss, t_typess_p, 1, neg_t.shape[0], device)
            pos = (pos_s_out * pos_t_out).sum(dim=-1)
            neg = (neg_s_out * neg_t_out).sum(dim=-1)
            batch_out = torch.cat((pos, neg)).sigmoid()
            y_true = torch.cat((torch.ones(batch_train.shape[0], dtype=int), torch.zeros(batch_train.shape[0], dtype=int)))
            loss = loss_fcn(batch_out, y_true.float().to(device))
            loss_avg.append(loss.item())
            auc, precision = accuracy(batch_out.to('cpu'), y_true.to('cpu'))
            auc_avg.append(auc)
            precision_avg.append(precision)
            loss.backward()
            optimizer.step()
            end = time.perf_counter()
            print('Epoch : {}, Step : {}, loss : {:.4f}, auc : {:.4f}, precision : {:.4f}, Running time: {:.4f} Seconds'.
                  format(run, step, loss.item(), auc, precision, end - start))

            if step % args.val_epochs == 0 and step != 0:
                with torch.no_grad():
                    model.eval()
                    ## Test
                    start = time.perf_counter()
                    test_loader = torch.utils.data.DataLoader(torch.LongTensor([_ for _ in range(test_links.shape[1])]), batch_size=5000, shuffle=False, drop_last=False)
                    test_out = torch.FloatTensor([]).to(device)
                    for batch_test in test_loader:
                        pos_s = test_links[0][batch_test]
                        pos_t = test_links[1][batch_test]

                        s_idxs, t_idxs, weightss = get_model_need(len(metapaths_b), sim_matrixs_b, t_typess_b, pos_s)
                        pos_s_out = model(features, features[s_type_b][pos_s], s_idxs, t_idxs, weightss, t_typess_b, 0, pos_s.shape[0], device)
                        s_idxs, t_idxs, weightss = get_model_need(len(metapaths_p), sim_matrixs_p, t_typess_p, pos_t)
                        pos_t_out = model(features, features[s_type_p][pos_t], s_idxs, t_idxs, weightss, t_typess_p, 1, pos_t.shape[0], device)
                        pos = (pos_s_out * pos_t_out).sum(dim=-1)

                        batch_out = pos.sigmoid()
                        test_out = torch.cat((test_out, batch_out), dim=0)

                    y_true = labels
                    auc, precision = accuracy(test_out.to('cpu'), y_true)
                    end = time.perf_counter()
                    print('Test auc : {:.4f}, precision : {:.4f}, Time : {:.4f}'.format(auc, precision, end - start))