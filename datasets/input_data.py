import scipy.sparse as sp
import scipy.io
import inspect
from torch.utils.data import Dataset
import networkx as nx
import logging
import numpy as np

logger = logging.getLogger(__name__)
class LoadData(Dataset):
    def __init__(self,data_source):
        if data_source in ['BlogCatalog','Flickr']:
            adj, features, labels = self.load_AN(data_source)
        elif data_source in ['ACM']:
            adj, features, labels = self.load_ACM(data_source)
        labels = np.squeeze(labels)
        adj_ = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj_ =adj_ + sp.eye(adj_.shape[0])
        self.graph = nx.from_scipy_sparse_matrix(adj_, create_using=nx.DiGraph())
        num_nodes = adj.shape[0]
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        # Some preprocessing
        if data_source in ['BlogCatalog','Flickr','ACM']:
            features_ = sparse_to_tuple(features.tocoo())
            num_features = features_[2][1]
            features_nonzero = features_[1].shape[0]
            features = np.asarray(features.todense())
        else:
            num_features = features.shape[1]
            features_nonzero = num_features
        self.features_nonzero = features_nonzero
        self.adj_label = adj_label
        self.features = features
        self.labels = labels
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.num_labels = np.max(labels)
    #
    def load_ACM(self, data_source):
        data = scipy.io.loadmat("./data/dataset/{}.mat".format(data_source))
        labels = data["Label"]

        attr_ = data["Features"]
        attributes = sp.csr_matrix(attr_)
        network = sp.lil_matrix(data["Network"])
        return network, attributes, labels

    def load_data(self,data_source):
        data = scipy.io.loadmat("./data/dataset/{}.mat".format(data_source))
        labels = data["Label"]

        attr_ = data["Attributes"]
        attributes = sp.csr_matrix(attr_)
        network = sp.lil_matrix(data["Network"])

        return network, attributes, labels
    def read_label(self,inputFileName):
        f = open(inputFileName, "r")
        lines = f.readlines()
        f.close()
        N = len(lines)
        y = np.zeros(N, dtype=int)
        i = 0
        for line in lines:
            l = line.strip("\n\r")
            y[i] = int(l)
            i += 1
        return y

    def load_AN(self,dataset):
        edge_file = open("./data/{}.edge".format(dataset), 'r')
        attri_file = open("./data/{}.node".format(dataset), 'r')
        label_file = "./data/" + dataset + ".label"
        y = self.read_label(label_file)

        edges = edge_file.readlines()
        attributes = attri_file.readlines()
        node_num = int(edges[0].split('\t')[1].strip())
        edge_num = int(edges[1].split('\t')[1].strip())
        attribute_number = int(attributes[1].split('\t')[1].strip())
        print("dataset:{}, node_num:{},edge_num:{},attribute_num:{}".format(dataset, node_num, edge_num,
                                                                            attribute_number))
        edges.pop(0)
        edges.pop(0)
        attributes.pop(0)
        attributes.pop(0)
        adj_row = []
        adj_col = []

        for line in edges:
            node1 = int(line.split('\t')[0].strip())
            node2 = int(line.split('\t')[1].strip())
            adj_row.append(node1)
            adj_col.append(node2)
        adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))

        att_row = []
        att_col = []
        for line in attributes:
            node1 = int(line.split('\t')[0].strip())
            attribute1 = int(line.split('\t')[1].strip())
            att_row.append(node1)
            att_col.append(attribute1)
        attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))
        return adj, attribute, y





def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape





