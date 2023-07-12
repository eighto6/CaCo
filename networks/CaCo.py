import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch.autograd import Variable
class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def forward(self, z, activation=None):
        """Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())

        return activation(adj)

class CaCo(nn.Module):
    def __init__(self,
                 g,
                 n_samples,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(CaCo, self).__init__()
        self.g = g
        self.n = n_layers
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.mean = nn.ModuleList()
        self.std = nn.ModuleList()
        # input layer
        self.mean.append(GraphConv(in_feats, n_hidden, bias=False, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.mean.append(GraphConv(n_hidden, n_hidden, bias=False, activation=activation))
        # output layer
        self.mean.append(GraphConv(n_hidden, n_classes, bias=False))
        self.std = GraphConv(n_hidden, n_classes, bias=False)

        self.linear_Za = torch.nn.Linear(n_classes,n_classes,bias=False)
        self.linear_Zb = torch.nn.Linear(n_classes, n_classes, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,g,adj,features):
        h = features
        for i, layer in enumerate(self.mean):
            if i!= 0:
                h = self.dropout(h)
            if i == self.n:
                self.A_mean = layer(adj,h)
                self.A_std = self.std(adj,h)
            h = layer(adj, h)
        k = features
        for i, layer in enumerate(self.mean):
            if i!= 0:
                k = self.dropout(k)
            if i == self.n:
                self.S_mean = layer(g,k)
                self.S_std = self.std(g,k)
            k = layer(g, k)

        if features.is_cuda:
            self.A_z = self.A_mean + torch.randn([self.n_samples, self.n_classes]).cuda() * torch.exp(self.A_std).cuda()
            self.S_z = self.S_mean + torch.randn([self.n_samples, self.n_classes]).cuda() * torch.exp(self.S_std).cuda()
        else:
           self.A_z = self.A_mean + torch.randn([self.n_samples, self.n_classes]).cpu() * torch.exp(self.A_std).cpu()
           self.S_z = self.S_mean + torch.randn([self.n_samples, self.n_classes]).cpu() * torch.exp(self.S_std).cpu()

        #
        self.z1 = (self.A_z -self.A_z.mean(0))/self.A_z.std(0)
        self.z2 = (self.S_z -self.S_z.mean(0))/self.S_z.std(0)
        self.z1 = self.linear_Za(self.z1)
        self.z2 = self.linear_Zb(self.z2)
        return self.z1,self.z2,self.A_mean,self.S_mean,self.A_std,self.S_std
    def get_embedding(self,graph,x):
        h = x
        for i, layer in enumerate(self.mean):
            if i != 0:
                h = self.dropout(h)
            if i == self.n:
                self.means = layer(graph, h)
                self.stds = self.std(graph, h)
            h = layer(graph, h)
       # self.embedding = self.means + torch.randn([self.n_samples, self.n_classes]).cpu() * torch.exp(
        #        self.stds).cpu()
       # self.embedding = self.means + torch.randn([self.n_samples, self.n_classes]).cuda() * torch.exp(self.stds).cuda()
        return h
class MLP(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_hidden2,
                 n_layers,
                 activation,
                 dropout):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            nn.Sequential(nn.Linear(in_feats, n_hidden),
                          nn.ReLU(inplace=True)))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(nn.Linear(n_hidden, n_hidden),
                              nn.ReLU(inplace=True)))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_hidden2))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
        return h

