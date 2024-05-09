import numpy as np
import scipy.sparse as sp
import torch
import dgl
from collections import Counter
from sklearn.metrics import f1_score
import random

def get_node_id_pubmed(id):
    num_gene = 13561
    num_disease = 20163
    num_chemical = 26522
    num_species = 2863
    idx_disease = num_gene
    idx_chemical = num_gene + num_disease
    idx_species = num_gene + num_disease + num_chemical
    num_nodes = num_gene + num_disease + num_chemical + num_species

    id = int(id)
    if id < num_gene:
        return id
    elif id < num_gene + num_disease:
        return (id - idx_disease)
    elif id < num_gene + num_disease + num_chemical:
        return (id - idx_chemical)
    else:
        return (id - idx_species)

def load_PubMed(data_path, data_name, is_normalize):
    num_gene = 13561
    num_disease = 20163
    num_chemical = 26522
    num_species = 2863
    idx_disease = num_gene
    idx_chemical = num_gene + num_disease
    idx_species = num_gene + num_disease + num_chemical
    num_nodes = num_gene + num_disease + num_chemical + num_species

    path = data_path + data_name + '/'
    feature_file = 'node.dat'
    link_file = 'link.dat'
    newid_file = 'new_id.dat'
    label_train_file = 'label.dat'
    label_test_file = 'label.dat.test'

    newid = np.zeros(num_nodes).astype(int)
    with open(path + newid_file) as f:
        line_data = f.readline()
        while (line_data):
            o_id, n_id, _= line_data.split()
            o_id = int(o_id)
            n_id = int(n_id)
            newid[o_id] = n_id
            line_data = f.readline()

    labels = torch.zeros(num_disease, dtype=int) - 1
    train_idx = []
    test_idx = []
    with open(path + label_train_file) as f:
        line_data = f.readline()
        while (line_data):
            o_id, _, _, label = line_data.split()
            idx = newid[int(o_id)] - idx_disease
            train_idx.append(idx)
            labels[idx] = int(label)
            line_data = f.readline()

    with open(path + label_test_file) as f:
        line_data = f.readline()
        while (line_data):
            o_id, _, _, label = line_data.split()
            idx = newid[int(o_id)] - idx_disease
            test_idx.append(idx)
            labels[idx] = int(label)
            line_data = f.readline()

    edge_s = {}
    edge_t = {}
    for i in range(10):
        edge_s[i] = []
        edge_t[i] = []
    with open(path + link_file) as f:
        line_data = f.readline()
        while(line_data):
            s_id, t_id, link_type, _ = line_data.split()
            s_id = get_node_id_pubmed(newid[int(s_id)])
            t_id = get_node_id_pubmed(newid[int(t_id)])
            edge_s[int(link_type)].append(s_id)
            edge_t[int(link_type)].append(t_id)
            line_data = f.readline()

    g = dgl.heterograph({
        ('gene', 'gag', 'gene'): (torch.tensor(edge_s[0]), torch.tensor(edge_t[0])),
        ('gene', 'gcd', 'disease'): (torch.tensor(edge_s[1]), torch.tensor(edge_t[1])),
        ('disease', 'dad', 'disease'): (torch.tensor(edge_s[2]), torch.tensor(edge_t[2])),
        ('chemical', 'cig', 'gene'): (torch.tensor(edge_s[3]), torch.tensor(edge_t[3])),
        ('chemical', 'cid', 'disease'): (torch.tensor(edge_s[4]), torch.tensor(edge_t[4])),
        ('chemical', 'cac', 'chemical'): (torch.tensor(edge_s[5]), torch.tensor(edge_t[5])),
        ('chemical', 'cis', 'species'): (torch.tensor(edge_s[6]), torch.tensor(edge_t[6])),
        ('species', 'swg', 'gene'): (torch.tensor(edge_s[7]), torch.tensor(edge_t[7])),
        ('species', 'swd', 'disease'): (torch.tensor(edge_s[8]), torch.tensor(edge_t[8])),
        ('species', 'sas', 'species'): (torch.tensor(edge_s[9]), torch.tensor(edge_t[9]))
    })
    new_edges = {}
    ntypes = set()
    for etype in g.etypes:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    features_all = torch.zeros((num_nodes, 200))
    with open(path + feature_file, encoding='utf-8') as f:
        line_data = f.readline()
        while (line_data):
            node_id, _, _, feature = line_data.split()
            node_id = newid[int(node_id)]
            feature = torch.FloatTensor(list(map(float, feature.split(','))))
            features_all[node_id] = feature
            line_data = f.readline()
    print("Done Feature!")
    if is_normalize:
        features_all = features_all / torch.sum(features_all, dim=1, keepdim=True)

    ## ['chemical', 'disease', 'gene', 'species']
    features = {}
    features[0] = features_all[idx_chemical:idx_species]
    features[1] = features_all[idx_disease:idx_chemical]
    features[2] = features_all[:num_gene]
    features[3] = features_all[idx_species:]

    return new_g, features, torch.LongTensor(labels).unsqueeze(1), torch.LongTensor(train_idx), torch.LongTensor(test_idx)

def get_node_id_yelp(id):
    num_business = 7474
    num_location = 39
    num_stars = 9
    num_phrase = 74943
    idx_loaction = num_business
    idx_stars = num_business + num_location
    idx_phrase = num_business + num_location + num_stars
    num_nodes = num_business + num_location + num_stars + num_phrase

    id = int(id)
    if id < num_business:
        return id
    elif id < num_business + num_location:
        return (id - idx_loaction)
    elif id < num_business + num_location + num_stars:
        return (id - idx_stars)
    else:
        return (id - idx_phrase)

def load_Yelp(data_path, data_name, is_normalize):
    num_business = 7474
    num_location = 39
    num_stars = 9
    num_phrase = 74943
    idx_loaction = num_business
    idx_stars = num_business + num_location
    idx_phrase = num_business + num_location + num_stars
    num_nodes = num_business + num_location + num_stars + num_phrase

    path = data_path + data_name + '/'
    feature_file = 'features.npy'
    link_file = 'link.dat'
    label_train_file = 'label.dat'
    label_test_file = 'label.dat.test'
    newid_file = 'new_id.dat'

    newid = np.zeros(num_nodes).astype(int)
    with open(path + newid_file) as f:
        line_data = f.readline()
        while (line_data):
            o_id, n_id = line_data.split()
            o_id = int(o_id)
            n_id = int(n_id)
            newid[o_id] = n_id
            line_data = f.readline()

    labels = torch.zeros((num_business, 16), dtype=int)
    train_idx = []
    test_idx = []
    with open(path + label_train_file) as f:
        line_data = f.readline()
        while (line_data):
            o_id, _, _, label = line_data.split()
            idx = newid[int(o_id)]
            train_idx.append(idx)
            label = label.split(',')
            for label_temp in label:
                labels[idx][int(label_temp)] = 1
            line_data = f.readline()

    with open(path + label_test_file) as f:
        line_data = f.readline()
        while (line_data):
            o_id, _, _, label = line_data.split()
            idx = newid[int(o_id)]
            test_idx.append(idx)
            label = label.split(',')
            for label_temp in label:
                labels[idx][int(label_temp)] = 1
            line_data = f.readline()

    edge_s = {}
    edge_t = {}
    for i in range(4):
        edge_s[i] = []
        edge_t[i] = []
    with open(path + link_file, encoding='utf-8') as f:
        line_data = f.readline()
        while(line_data):
            s_id, t_id, link_type, _ = line_data.split()
            s_id = get_node_id_yelp(newid[int(s_id)])
            t_id = get_node_id_yelp(newid[int(t_id)])
            edge_s[int(link_type)].append(s_id)
            edge_t[int(link_type)].append(t_id)
            line_data = f.readline()

    g = dgl.heterograph({
        ('business', 'locatedin', 'location'): (torch.tensor(edge_s[0]), torch.tensor(edge_t[0])),
        ('business', 'rate', 'stars'): (torch.tensor(edge_s[1]), torch.tensor(edge_t[1])),
        ('business', 'describedwith', 'phrase'): (torch.tensor(edge_s[2]), torch.tensor(edge_t[2])),
        ('phrase', 'context', 'phrase'): (torch.tensor(edge_s[3]), torch.tensor(edge_t[3])),
    })
    new_edges = {}
    ntypes = set()
    for etype in g.etypes:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    features_all = torch.zeros((num_nodes, 200))
    feature_temp = torch.FloatTensor(np.load(path + feature_file))
    for i in range(features_all.shape[0]):
        id_temp = newid[i]
        features_all[i] = feature_temp[id_temp]
    print("Done Feature!")
    if is_normalize:
        features_all = features_all / torch.sum(features_all, dim=1, keepdim=True)

    ## ['business', 'location', 'phrase', 'stars']
    features = {}
    features[0] = features_all[:num_business]
    features[1] = features_all[idx_loaction:idx_stars]
    features[2] = features_all[idx_phrase:]
    features[3] = features_all[idx_stars:idx_phrase]

    return new_g, features, torch.LongTensor(labels), torch.LongTensor(train_idx), torch.LongTensor(test_idx)

def get_node_id_dblp(id):
    num_phrase = 217557
    num_author = 1766361
    num_venue = 5076
    num_year = 83
    idx_author = num_phrase
    idx_venue = num_phrase + num_author
    idx_year = num_phrase + num_author + num_venue
    num_nodes = num_phrase + num_author + num_venue + num_year

    id = int(id)
    if id < num_phrase:
        return id
    elif id < num_phrase + num_author:
        return (id - idx_author)
    elif id < num_phrase + num_author + num_venue:
        return (id - idx_venue)
    else:
        return (id - idx_year)

def load_dblp(data_path, data_name, is_normalize):
    num_phrase = 217557
    num_author = 1766361
    num_venue = 5076
    num_year = 83
    idx_author = num_phrase
    idx_venue = num_phrase + num_author
    idx_year = num_phrase + num_author + num_venue
    num_nodes = num_phrase + num_author + num_venue + num_year

    path = data_path + data_name + '/'
    feature_file = 'node.dat'
    link_file = 'link.dat'
    label_train_file = 'label.dat'
    label_test_file = 'label.dat.test'

    labels = torch.zeros(num_author, dtype=int) - 1
    train_idx = []
    test_idx = []
    with open(path + label_train_file) as f:
        line_data = f.readline()
        while (line_data):
            o_id, _, _, label = line_data.split()
            idx = int(o_id) - idx_author
            train_idx.append(idx)
            labels[idx] = int(label)
            line_data = f.readline()

    with open(path + label_test_file) as f:
        line_data = f.readline()
        while (line_data):
            o_id, _, _, label = line_data.split()
            idx = int(o_id) - idx_author
            test_idx.append(idx)
            labels[idx] = int(label)
            line_data = f.readline()

    edge_s = {}
    edge_t = {}
    for i in range(6):
        edge_s[i] = []
        edge_t[i] = []
    with open(path + link_file, encoding='utf-8') as f:
        line_data = f.readline()
        while(line_data):
            s_id, t_id, link_type, _ = line_data.split()
            s_id = get_node_id_dblp(int(s_id))
            t_id = get_node_id_dblp(int(t_id))
            edge_s[int(link_type)].append(s_id)
            edge_t[int(link_type)].append(t_id)
            line_data = f.readline()

    g = dgl.heterograph({
        ('phrase', 'cooccur', 'phrase'): (torch.tensor(edge_s[0]), torch.tensor(edge_t[0])),
        ('author', 'coauthor', 'author'): (torch.tensor(edge_s[1]), torch.tensor(edge_t[1])),
        ('author', 'cite', 'author'): (torch.tensor(edge_s[2]), torch.tensor(edge_t[2])),
        ('author', 'study', 'phrase'): (torch.tensor(edge_s[3]), torch.tensor(edge_t[3])),
        ('author', 'publishin', 'venue'): (torch.tensor(edge_s[4]), torch.tensor(edge_t[4])),
        ('author', 'activein', 'year'): (torch.tensor(edge_s[5]), torch.tensor(edge_t[5])),
    })
    new_edges = {}
    ntypes = set()
    for etype in g.etypes:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    features_all = torch.zeros((num_nodes, 300))
    with open(path + feature_file, encoding='utf-8') as f:
        line_data = f.readline()
        while (line_data):
            node_id, _, _, feature = line_data.split()
            node_id = int(node_id)
            feature = torch.FloatTensor(list(map(float, feature.split(','))))
            features_all[node_id] = feature
            line_data = f.readline()
    print("Done Feature!")
    if is_normalize:
        features_all = features_all / torch.sum(features_all, dim=1, keepdim=True)

    features = {}
    features[0] = features_all[idx_author:idx_venue]
    features[1] = features_all[:num_phrase]
    features[2] = features_all[idx_venue:idx_year]
    features[3] = features_all[idx_year:]

    return new_g, features, torch.LongTensor(labels).unsqueeze(1), torch.LongTensor(train_idx), torch.LongTensor(test_idx)

def random_walk_sim(batch_idx, g, metapath, num_per_node, K, random_flag):
    list_idx = list(batch_idx)
    list_idx = [val for val in list_idx for _ in range(num_per_node)]
    walks, types = dgl.sampling.random_walk(g=g, nodes=list_idx, metapath=metapath)
    s_type = types[0]
    num_s = g.num_nodes(g.ntypes[types[0]])
    tnode_types = set(types[1:].tolist()) # remove source node type
    row_nodes = {}
    col_nodes = {}
    topk_counts = {}
    sim_mitrix = {}

    for t_type in tnode_types:
        row_nodes[t_type] = []
        col_nodes[t_type] = []
        topk_counts[t_type] = []

    if random_flag:
        for i in range(batch_idx.shape[0]):
            s_node = int(walks[i * num_per_node, 0])
            for t_type in tnode_types:
                t_indexs = torch.nonzero(types == t_type)

                if t_type == types[0]:
                    ## delete source node:
                    t_indexs = t_indexs[1:]

                t_nodes = []
                for t_index in t_indexs:
                    t_node = walks[i * num_per_node:(i + 1) * num_per_node, t_index].squeeze().tolist()
                    t_nodes.extend(t_node)

                t_nodes = list(filter(lambda x: x != -1, t_nodes))
                if len(t_nodes) != 0:
                    if len(t_nodes) >= K:
                        topk_node = random.sample(t_nodes, K)
                    else:
                        topk_node = t_nodes
                    topk_count = [1 for _ in topk_node]
                    row_nodes[t_type].extend([s_node for _ in range(len(topk_node))])
                    col_nodes[t_type].extend(topk_node)
                    topk_counts[t_type].extend(topk_count / np.sum(topk_count))
    else:
        for i in range(batch_idx.shape[0]):
            s_node = int(walks[i * num_per_node, 0])
            for t_type in tnode_types:
                t_indexs = torch.nonzero(types == t_type)

                if t_type == types[0]:
                    ## delete source node:
                    t_indexs = t_indexs[1:]

                t_nodes = []
                for t_index in t_indexs:
                    t_node = walks[i * num_per_node:(i + 1) * num_per_node, t_index].squeeze().tolist()
                    t_nodes.extend(t_node)

                t_nodes = list(filter(lambda x: x != -1, t_nodes))
                if len(t_nodes) != 0:
                    topk = Counter(t_nodes).most_common(K)
                    topk_node = [_[0] for _ in topk]
                    topk_count = [_[1] for _ in topk]
                    row_nodes[t_type].extend([s_node for _ in range(len(topk_node))])
                    col_nodes[t_type].extend(topk_node)
                    topk_counts[t_type].extend(topk_count / np.sum(topk_count))

    for t_type in tnode_types:
        num_t = g.num_nodes(g.ntypes[t_type])
        sim_mitrix[t_type] = sp.csr_matrix((topk_counts[t_type], (row_nodes[t_type], col_nodes[t_type])), shape=(num_s, num_t))

    return sim_mitrix, list(tnode_types), int(s_type)

def get_weights_sidx(sim_matrix, idx):
    sim_matrix = sim_matrix[idx]
    s_idx, t_idx = sim_matrix.nonzero()
    s_idx = torch.LongTensor(s_idx)
    weights = torch.FloatTensor(sim_matrix.data)
    return s_idx, t_idx, weights

def get_model_need(n_metapaths, sim_matrixs, t_typess, batch):
    s_idxs = []
    t_idxs = []
    weightss = []
    for i in range(n_metapaths):
        sim_matrix = sim_matrixs[i]
        t_types = t_typess[i]
        s_idx = {}
        t_idx = {}
        weights = {}
        for t_type in t_types:
            s_idx[t_type], t_idx[t_type], weights[t_type] = get_weights_sidx(sim_matrix[t_type], batch)
        s_idxs.append(s_idx)
        t_idxs.append(t_idx)
        weightss.append(weights)
    return  s_idxs, t_idxs, weightss

def graph2matrix(src, dst, matrix_temp):
    """Convert dgl edge to adj"""
    for i in range(len(src)):
        if src[i] != dst[i]:
            matrix_temp[src[i], dst[i]] = 1
    adj = sp.csr_matrix(matrix_temp)
    return adj

def trans_sparse_matrix(D):
    x = sp.find(D)
    return sp.csc_matrix((x[2], (x[1], x[0])), shape = (D.shape[1], D.shape[0]))

def accuracy(output, labels, dataset):
    if dataset == 'Yelp':
        macro_f1 = []
        micro_f1 = []
        output = np.int64(output.cpu() > 0.5)
        labels = labels.cpu()
        for i in range(labels.shape[0]):
            preds = output[i]
            correct = labels[i]
            macro_f1.append(f1_score(correct, preds, average='macro'))
            micro_f1.append(f1_score(correct, preds, average='micro'))
        macro_f1 = np.array(macro_f1).mean()
        micro_f1 = np.array(micro_f1).mean()
    else:
        preds = output.max(dim=1)[1].view(-1, 1).cpu()
        correct = labels.cpu()
        macro_f1 = f1_score(correct, preds, average='macro')
        micro_f1 = f1_score(correct, preds, average='micro')
    return macro_f1, micro_f1











