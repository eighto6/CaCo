from dgl.data import load_data
from dgl import DGLGraph
import torch.utils.data
import numpy as np
import torch
import scipy.sparse as sp
import networkx as nx
from datasets.prepocessing import one_class_processing
from datasets.input_data import LoadData
def loader(args):
    # load and preprocess dataset
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        data = load_data(args)
    else:
        data = LoadData(args.dataset)
    print(f'normal_class is {args.normal_class}')

    labels,train_mask,val_mask,test_mask=one_class_processing(data,args.normal_class,args)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    num = labels[labels==0]
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(labels)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    n_nodes = features.shape[0]



    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()


    # graph preprocess and calculate normalization factor
    g = data.graph
    idx = np.arange(len(labels))
    adj = sp.coo_matrix((np.ones(labels.shape[0]),
                         (idx, idx)),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    g_cnn = nx.from_scipy_sparse_matrix(adj, create_using=nx.DiGraph())
    # add self loop
    if args.self_loop:
        g.remove_edges_from(nx.selfloop_edges(g))
        g.add_edges_from(zip(g.nodes(), g.nodes()))
        g_cnn.add_edges_from(zip(g.nodes(), g.nodes()))
    if cuda:
        g = DGLGraph(g).to('cuda:{}'.format(args.gpu))
        g_cnn = DGLGraph(g_cnn).to('cuda:{}'.format(args.gpu))
    else:
        g = DGLGraph(g)
        g_cnn = DGLGraph(g_cnn)
    n_edges = g.number_of_edges()
    if args.norm:

        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        if cuda:
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)



    datadict={'g':g,'g_cnn':g_cnn,'features':features,'labels':labels,'train_mask':train_mask,
        'val_mask':val_mask,'test_mask': test_mask,'input_dim':in_feats,'n_classes':n_classes,'n_edges':n_edges,
              'num_node':n_nodes}


    return datadict

