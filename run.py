# coding=utf-8
import argparse
import logging
import os
import random
from time import strftime, localtime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from datasets import not_split_load_dataseta_and_vocabs
from model_rel import Rel_Overall
from trainer import train_overall

# logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    log_file = '/logs/{}.{}.log'.format(args.dataset_name, strftime("%y%m%d.%H%M", localtime()))
    fh = logging.FileHandler(args.base_dir + log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # logger.addHandler(ch)
    logger.addHandler(fh)
    args.log_file = log_file

    return logger

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--log_file', default='/logs/log_1115.log')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--multi_gpu', default=True)
    parser.add_argument('--devices', default=[0,1])
    parser.add_argument('--dataset_name', type=str, default='mams',
                            choices=['rest', 'laptop', 'mams'], help='Choose absa dataset.')
    parser.add_argument('--output_dir', type=str, default='/home/niuhao/project/ASC/MAMS_Tagging/data/output-gcn',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint')
    # parser.add_argument('--sub_dir', type=str, default='/get_gaussian_case_for_predefined')
    parser.add_argument('--multi_sub_dir', type=str, default='/multi_aspect_chpt')
    
    parser.add_argument('--seed', type=int, default=8187,
                        help='random seed for initialization')
    parser.add_argument('--pure_bert', default=False)
    parser.add_argument('--bert_unfreeze', type=list, default=['layer.5', 'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'pooler', 'out'])
    #########################for disentangle GAT########################
    parser.add_argument('--is_lr_decay', type=bool, default=True)
    parser.add_argument('--lr_decay', default=1e-8, type=float, help='decline learning rate')
    parser.add_argument('--embedding_type', default='bert', type=str)
    
    parser.add_argument('--dep_embed', default='combine', choices=['bert', 'random', 'combine'])
    parser.add_argument('--deps_share_weight_with_nodes', default=True)
    # parser.add_argument('--single_disen_att_type', default=['AS_Wi', 'AS_DW', 'DA_W', 'DA_DW'])  #['AS_Wi', 'AS_DW', 'DA_W', 'DA_DW']
    parser.add_argument('--multi_disen_att_type', default=['AS_Wi', 'AS_DW', 'AS_TW'])  #'AS_Wi', 'AS_DW', 'AS_TW', 'AS_NTN'
    parser.add_argument('--implicit_edge', default=False)
    parser.add_argument('--num_node_type', default=2)
    
    #############for edge feature##############
    parser.add_argument('--g_encoder', default='multi_att', choices=['multi_att', 'graph_trans'])
    parser.add_argument('--inter_aspect_edge', default='SPN', choices=['composed', 'SPN'])
    parser.add_argument('--dep_tag_type', default='composed', choices=['composed', 'n_conn'])
    parser.add_argument('--softmax_first', default=False, help='do softmax before att_combine or after')
    parser.add_argument('--wi_use_ntn', default=False)

    ########rgat###############
    parser.add_argument('--opn', default='corr', choices=['corr', 'sub', 'mult', 'add', 'cconv'])

    # Model parameters
    parser.add_argument('--glove_dir', type=str, default='/data1/SHENWZH/wordvec',
                        help='Directory storing glove embeddings')
    parser.add_argument('--bert_model_dir', type=str, default='bert-base-uncased',
                        help='Path to pre-trained Bert model.')
    # parser.add_argument('--gat_bert', action='store_true', default=True,
                        # help='Cat text and aspect, [cls] to predict.')

    parser.add_argument('--highway', action='store_true', default=False,
                        help='Use highway embed.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm or highway or elmo.')


    parser.add_argument('--add_non_connect',  type= bool, default=True,
                        help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
    parser.add_argument('--multi_hop',  type= bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type = int, default=4,   #4
                        help='max number of hops')


    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')
    
    parser.add_argument('--dropout', type=float, default=0.3,    #0.3
                        help='Dropout rate for embedding.')

    parser.add_argument('--gcn_dropout', type=float, default=0.2,   #0.3
                        help='Dropout rate for GCN.')

    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')

    parser.add_argument('--hidden_size', type=int, default=200,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=128,   #300
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")  #8
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")  #16
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--weight_decay", default=1e-8, type=float,     #1e-6   1e-8
                        help="Weight deay if we apply some.")   #0.0
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    
    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(args))
        


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    # Parse args
    args = parse_args()
    args.base_dir = os.path.dirname(os.path.abspath(__file__))
    logger = set_logger(args)
    args.logger = logger
    # check_args(args)

    # Setup CUDA, GPU training
    if args.gpu != '-1' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    total_rest = []
    set_seed(args)

    logger.info(vars(args))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
    args.tokenizer = tokenizer

    # Load datasets and vocabs
    overall_train_dataset, overall_test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab = not_split_load_dataseta_and_vocabs(args)

    # Build Model
    model = Rel_Overall(args, dep_tag_vocab, pos_tag_vocab['len'])
    if args.multi_gpu:
        model = nn.DataParallel(model, device_ids=args.devices).to(args.device)
        
    
    # Train
    model.to(args.device)
    _, _,  all_eval_results = train_overall(args, overall_train_dataset, overall_test_dataset, model)

    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x['acc'])
        best_eval_result['seed'] = args.seed
        total_rest += [best_eval_result]
        args.logger.info(total_rest)
        for key in sorted(best_eval_result.keys()):
            args.logger.info("  %s = %s", key, str(best_eval_result[key]))
    
    acc_list, f1_list = [],[]
    for item in total_rest:
        acc_list += [item['acc']]
        f1_list += [item['f1']]
    acc_list = np.array(acc_list)
    f1_list = np.array(f1_list)
    acc_mean = np.mean(acc_list)
    f1_neam = np.mean(f1_list)
    acc_std = np.std(acc_list)
    f1_std = np.std(f1_list)
    result = {'acc_mean': acc_mean, 'acc_std': acc_std, 'f1_neam': f1_neam, 'f1_std': f1_std}
    for key in result.keys():
        args.logger.info(" %s = %s ", key, str(result[key]))



if __name__ == "__main__":
    main()

