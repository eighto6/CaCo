import torch.nn.functional as F
import torch
from networks.CaCo import CaCo

def init_model(args,input_dim,input_sdim):

    if args.module== 'CaCo':
        model = CaCo(None,
                input_sdim,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout)
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    if cuda:
        device = torch.device("cuda:{}".format(args.gpu))
        model.to(device)

    print(f'Parameter number of {args.module} Net is: {count_parameters(model)}')

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)