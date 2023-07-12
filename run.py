import argparse
from dgl.data import register_data_args
import logging
from optim import multrainer
from optim.loss import loss_function, init_center
from datasets import dataloader
from networks.init import init_model
import numpy as np
import torch
from dgl import random as dr
import os


def main(args):
    if args.seed != -1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # torch.backends.cudnn.deterministic=True
        dr.seed(args.seed)

    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')
    checkpoints_path = f'./checkpoints/{args.dataset}-{args.module}-{args.lamda}+bestcheckpoint.pt'
    logging.basicConfig(filename=f"./log/{args.dataset}-{args.module}-{args.lamda}.log", filemode="a",
                        format="%(asctime)s-%(name)s-%(levelname)s-%(message)s", level=logging.INFO)
    logger = logging.getLogger('Caco')
    data = dataloader.loader(args)
    model = init_model(args, data['input_dim'], len(data['features']))
    model = multrainer.train(args, logger, data, model, checkpoints_path)

    return model


DATASETS_NAME = {
  'cora': 7,
  'citeseer': 6,
  'pubmed': 3,
  'BlogCatalog':6,
  'Flickr':9,
   'ACM':9,

}
SEEDS = [
  4,8,16,32,52
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCAADG')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--lamda", type=float, default=0.2,
                        help="Weight for attribute")
    parser.add_argument("--seed", type=int, default=52,
                        help="random seed, -1 means dont fix seed")
    parser.add_argument("--module", type=str, default='CaCo',
                        help="CaCo")
    parser.add_argument('--n-worker', type=int, default=1,
                        help='number of workers when dataloading')

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--normal-class", type=int, default=0,
                        help="normal class")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=64,
                        help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--norm", action='store_true',
                        help="graph normalization (default=False)")
    parser.add_argument("--task", type=str, default='AD',
                        help="AD/C")
    parser.add_argument("--abclass", type=int, )
    parser.add_argument("--p1", type=int, default=90)
    parser.add_argument("--p2", type=int, default=80)
    parser.set_defaults(self_loop=True)
    parser.set_defaults(norm=False)
    args = parser.parse_args()
    if args.gpu>0:
        os.environ['CUDA_VISIBLE_DEVICE'] = '{}'.format(args.gpu)
    print('model: {}'.format(args.module))
    for dataset_name in list(DATASETS_NAME.keys()):
        args.dataset = dataset_name
        results_dir = 'Caco/{}/{}'.format(args.module, args.dataset)
        args.outf = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        file2print = open('{}/results_{}.log'.format(results_dir, args.module), 'a+')
        file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, args.module), 'a+')

        import datetime

        print(datetime.datetime.now())
        print(datetime.datetime.now(), file=file2print)
        print(datetime.datetime.now(), file=file2print_detail)
        print("Model\tDataset\tlayers\tembedding\tlamda\tNormal_Label\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch",
              file=file2print_detail)

        print("Model\tDataset\tlayers\tembedding\tlamda\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch")
        print("Model\tDataset\tlayers\tembedding\tlamda\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print)
        file2print.flush()
        file2print_detail.flush()
        AUCs = {}
        APs = {}
        MAX_EPOCHs = {}
        for normal_idx in range(DATASETS_NAME[dataset_name]):
            args.abclass = normal_idx
            if args.dataset in ['BlogCatalog', 'Flickr', 'ACM'] and args.datamode != 'AD':
                normal_idx += 1

            print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))

            MAX_EPOCHs_seed = {}
            AUCs_seed = {}
            APs_seed = {}
            for seed in SEEDS:
                # np.random.seed(args.seed ** 2)

                args.seed = seed
                args.normal_class = normal_idx

                args.name = "%s/%s" % (args.module, args.dataset)
                expr_dir = os.path.join(args.outf, args.name, 'train')
                test_dir = os.path.join(args.outf, args.name, 'test')

                if not os.path.isdir(expr_dir):
                    os.makedirs(expr_dir)
                if not os.path.isdir(test_dir):
                    os.makedirs(test_dir)

                args1 = vars(args)
                file_name = os.path.join(expr_dir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args1.items()):
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
                    opt_file.write('-------------- End ----------------\n')

                print(args1)

                print("################", dataset_name, "##################")
                print("################  Train  ##################")
                res = main(args)
                auc_test = res[0]
                ap_test = res[1]
                epoch_max_point = res[2]
                AUCs_seed[seed] = auc_test
                APs_seed[seed] = ap_test
                MAX_EPOCHs_seed[seed] = epoch_max_point

            # End For

            MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
            AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
            AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
            APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
            APs_seed_std = round(np.std(list(APs_seed.values())), 4)

            print("Dataset: {} \t Normal Label: {} \t AUCs={}+{} \t APs={}+{} \t MAX_EPOCHs={}".format(
                dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
                MAX_EPOCHs_seed))

            print("{}\t{}\t{:.1f}\t{:.1f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
                args.module, dataset_name, args.n_layers, args.n_hidden,args.lamda, normal_idx, AUCs_seed_mean,
                AUCs_seed_std, APs_seed_mean,
                APs_seed_std, MAX_EPOCHs_seed_max
            ), file=file2print_detail)
            file2print_detail.flush()

            AUCs[normal_idx] = AUCs_seed_mean
            APs[normal_idx] = APs_seed_mean
            MAX_EPOCHs[normal_idx] = MAX_EPOCHs_seed_max

        print("{}\t{}\t{:.1f}\t{:.1f}\t{:.4f}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
            args.module, dataset_name, args.n_layers, args.n_hidden,args.lamda,
            np.mean(list(AUCs.values())),
            np.std(list(AUCs.values())),
            np.mean(list(APs.values())),
            np.std(list(APs.values())),
            np.max(list(MAX_EPOCHs.values()))),
            file=file2print)

        file2print.flush()

