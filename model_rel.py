import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

from model_gcn import *
from model_utils import *
from tree import *


class Rel_Overall(nn.Module):
    def __init__(self, args, dep_tag_vocab, pos_tag_num):
        super(Rel_Overall, self).__init__()
        self.args = args
        self.dep_tag_vocab = dep_tag_vocab
        dep_tag_num = len(dep_tag_vocab['stoi'])

        # Bert
        config = BertConfig.from_pretrained(args.bert_model_dir)
        self.bert = BertModel.from_pretrained(
            args.bert_model_dir, config=config, from_tf =False)
        if self.args.bert_unfreeze != []:   # if bert_unfreeze == [], all bert params are not frozen
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
                for pre_name in self.args.bert_unfreeze:
                    if pre_name in name:
                        param.requires_grad = True
                        break
        # if args.multi_gpu:
        #     self.bert = nn.DataParallel(self.bert, device_ids=args.devices).to(self.args.device)

        # self.dropout_bert = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(args.dropout)
        self.args.embedding_dim = config.hidden_size  # 768

        if args.highway:
            self.highway_dep = Highway(args.num_layers, self.args.embedding_dim)
            self.highway = Highway(args.num_layers, self.args.embedding_dim)

        ######################SPN##########################
        if self.args.inter_aspect_edge == 'SPN':
            hidden_dim = self.args.embedding_dim//2
            self.lstm = nn.LSTM(self.args.embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        ####################################Multi-aspect Disentangle GAT #####################################
        self.graph_encoder = nn.Sequential(*[Disen_GAT_For_Multi_Aspect(args, in_dim=self.args.embedding_dim, head_name=self.args.g_encoder).to(args.device) for i in range(args.num_heads)])

        self.fuse = GraphChannelAttLayer(num_channel=2)

        self.dep_embed = nn.Embedding(dep_tag_num+1, self.args.embedding_dim, padding_idx=0)
        # self.dep_embed.weight.requires_grad = False
        self.type_embed = nn.Embedding(self.args.num_node_type, self.args.embedding_dim)
        # self.type_embed.weight.requires_grad = False

        last_hidden_size = self.args.embedding_dim * 1
        # last_hidden_size = args.final_hidden_size*1
        layers = [
            nn.Linear(last_hidden_size, args.final_hidden_size), nn.ReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.ReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)

    def rel_transform(self, ent_embed, rel_embed):
        if   self.args.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
        elif self.args.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
        elif self.args.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
        else: raise NotImplementedError

        return trans_embed

    def forward(self, input_ids, input_aspect_ids, word_indexer, aspect_indexer, input_cat_ids, segment_ids, dep_tags, pos_class, text_len, aspect_len, dep_rels, dep_dirs, aspect_dep_tag_ids, SPN):  #, type_ids, shortest_path_ids
        fmask = (torch.ones_like(word_indexer) != word_indexer).float()  # (N，L)
        fmask[:,0] = 1   # fmask is to mask the field that beyond the sentence length

        input_cat_ids_re = input_cat_ids.view(-1, input_cat_ids.shape[-1])
        segment_ids_re = segment_ids.view(-1, segment_ids.shape[-1])
        outputs = self.bert(input_cat_ids_re, token_type_ids = segment_ids_re)
        feature_output = outputs[0] # (N, L, D)
        pool_out = outputs[1] #(N, D)

        aspect_feature  = feature_output * segment_ids_re.unsqueeze(2)
        aspect_feature = aspect_feature.sum(dim=1)
        # index select, back to original batched size.
        aspect_num = input_cat_ids.shape[1]
        word_indexer = word_indexer.unsqueeze(1)
        word_indexer_re = torch.repeat_interleave(word_indexer, aspect_num, dim=1)
        word_indexer_re = word_indexer_re.view(-1, word_indexer_re.shape[-1])
        feature = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_indexer_re)])
        ############################################################################################
        # dep_tags = dep_tags.view(-1, dep_tags.shape[-1])
        if self.args.dep_embed == 'bert':
            depshape = dep_tags.shape
            dep_tags = dep_tags.view(-1, depshape[-1])
            _, dep_feature = self.bert(dep_tags)
            dep_feature = dep_feature.view(depshape[0], depshape[1], -1, dep_feature.shape[-1])
        elif self.args.dep_embed == 'random':
            dep_feature = self.dep_embed(dep_tags)
        elif self.args.dep_embed == 'combine':
            dep_feature = self.dep_embed(dep_tags)  # combined reshaped dependency tags info
            dep_feature = dep_feature.sum(-2)
        if self.args.highway:
            dep_feature = self.highway_dep(dep_feature)
        
        ####################################Multi-aspect Disentangled GAT#######################################################
        feature_ = feature.view(input_cat_ids.shape[0], input_cat_ids.shape[1], -1, self.args.embedding_dim)
        aspect_feature = aspect_feature.view(input_cat_ids.shape[0], input_cat_ids.shape[1], -1)
        
        aspect_feature_ = aspect_feature.unsqueeze(1)
        aspect_feature_ = torch.repeat_interleave(aspect_feature_, aspect_num, dim=1)
        feature_ = torch.cat([feature_, aspect_feature_], dim=-2)

        ###################To get all dep_feature by adding SNP feature from bert tokens#################
        if self.args.inter_aspect_edge == 'SPN':
            SPN_shape = SPN.shape 
            SPN = SPN.view(SPN_shape[0], aspect_num, -1, SPN_shape[-1]).unsqueeze(-1)
            feature_output_ = feature_output.view(-1, aspect_num, feature_output.shape[-2], feature_output.shape[-1]).unsqueeze(1)
            feature_output_ = torch.repeat_interleave(feature_output_, aspect_num, dim=1)
            SPN_ = SPN * feature_output_

            SPN_ = SPN_.sum(-2)
        elif self.args.inter_aspect_edge == 'composed':
            tail_aspect_feature_ = aspect_feature_
            head_aspect_feature_ = aspect_feature.unsqueeze(2)
            head_aspect_feature_ = torch.repeat_interleave(head_aspect_feature_, aspect_num, dim=2)
            SPN_ = self.rel_transform(head_aspect_feature_, tail_aspect_feature_)
        dep_feature_ = torch.cat([dep_feature, SPN_], dim=-2)
        #####Here, aspect_dep_feature_ should be SPNs feature

        ####To generate Type feature####
        type_ids = torch.zeros(input_cat_ids.shape[0], input_cat_ids.shape[1], dep_feature.shape[2], dtype=torch.long).to(self.args.device)
        aspect_type_ids = torch.ones(input_cat_ids.shape[0], input_cat_ids.shape[1], input_cat_ids.shape[1], dtype=torch.long).to(self.args.device)
        type_feature = self.type_embed(type_ids)
        aspect_type_feature = self.type_embed(aspect_type_ids)
        all_type_feature = torch.cat([type_feature, aspect_type_feature], dim=-2)

        #######fmask add aspect_node mask, which is ones matrix########
        # fmask is also need to add aspect_node mask, and the aspect_node mask is ones matrix.
        aspect_mask = torch.ones(input_cat_ids.shape[0], aspect_num).to(self.args.device)
        all_fmask = torch.cat([fmask, aspect_mask], dim=-1)
        
        ################# graph encoder #################
        att_out = [g(feature_, dep_feature_, aspect_feature, all_type_feature, all_fmask) for g in self.graph_encoder]  # (N, 1, D) * num_heads
        feature_out = torch.cat(att_out, dim=-1)

        ################# sequence encoder #################
        B, A, _ = feature_out.shape
        sequence_out_a = pool_out.view(B, A,-1)
        
        feature_out = self.fuse([feature_out, sequence_out_a])
        x = self.dropout(feature_out)
        x_ = self.fcs(x)
        logit = self.fc_final(x_)
        
        return logit