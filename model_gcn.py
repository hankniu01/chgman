import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from comp import *


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)



class GraphChannelAttLayer(nn.Module):
    
    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        nn.init.constant_(self.weight, 1)  # equal weight

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        adj_list = adj_list.permute(1,0,2,3)
        batch_size = adj_list.shape[0]
        
        w = self.weight
        w = w.repeat(batch_size, 1, 1, 1)
        return torch.sum(adj_list * w, dim=1)

class Disen_GAT_For_Multi_Aspect(nn.Module):
    def __init__(self, args, in_dim, head_name, hidden_dim=128):
        super(Disen_GAT_For_Multi_Aspect, self).__init__()
        self.args = args
        self.head_name = head_name
        hidden_dim = in_dim // args.num_heads
        # hidden_dim = in_dim
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(in_dim, hidden_dim)
        self.key = nn.Linear(in_dim, hidden_dim)
        # self.key_comp = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(self.args.gcn_dropout)


        if 'AS_DW' in self.args.multi_disen_att_type:
            if self.args.implicit_edge == True:
                self.trans = nn.Linear(hidden_dim*2, hidden_dim)
                self.T1 = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim, hidden_dim))
                nn.init.xavier_normal_(self.T1.data)
                self.W1 = nn.Linear(hidden_dim*2, hidden_dim)

        self.attcombination = GraphChannelAttLayer(num_channel=len(self.args.multi_disen_att_type))
            
        if not self.args.deps_share_weight_with_nodes:
            self.value_type = nn.Linear(in_dim, hidden_dim)
            self.key_dep = nn.Linear(in_dim, hidden_dim)
            self.key_type = nn.Linear(in_dim, hidden_dim)
    
    def rel_transform(self, ent_embed, rel_embed):
        if   self.args.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
        elif   self.args.opn == 'cconv': 	trans_embed  = cconv(ent_embed, rel_embed)
        elif self.args.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
        elif self.args.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
        elif self.args.opn == 'add': 	trans_embed  = ent_embed + rel_embed
        else: raise NotImplementedError

        return trans_embed
    
    def calcu_gate(self, feat, gate1, gate2):
        
        G = gate1(feat)
        G = self.relu(G)
        G = gate2(G)
        G = G.squeeze(-1)
        G = torch.sigmoid(G)
        # G = torch.softmax(G, dim=-1)

        return G

    def type_att(self, AS_Q, T_K, scale_factor_att):
        
        AS_Q = AS_Q.unsqueeze(2)    # 8*5*1*128
        ###############aspect_to_type##########

        AS_TW_att = torch.matmul(AS_Q, T_K.transpose(-1, -2))

        return AS_TW_att
    
    def gate_head(self, W_V, AS_TW_att, dep_feature, all_type_feature, fmask):

        # control each node's contribuations wihtin one specific node type, divide into two flow
        ###############gate_z##################
        com_dep_feature = torch.cat([dep_feature, all_type_feature], dim=-1)
        e_gate = self.calcu_gate(com_dep_feature, self.gate1_DW, self.gate2_DW)   #B * aspect_num * num_node * embed_dim
        gate_a = AS_TW_att * e_gate

        gate_a = gate_a.transpose(-1, -2)
        fmask = fmask.unsqueeze(2)
        gate_a = mask_logits(gate_a, fmask)
        gate_a_prob = F.softmax(gate_a, dim=-1)  #[N, 1, L]
        gate_a_prob = self.dropout(gate_a_prob)
        gate_z = torch.matmul(gate_a_prob.transpose(-1, -2).unsqueeze(-2), W_V)
        gate_z = gate_z.squeeze(2)

        return gate_z
    
    def att_head(self, AS_Q, W_K, W_V, DW_K, T_V, AS_TW_att, scale_factor, fmask):
        
        # ['AS_Wi', 'AS_DW', 'AS_TW']
        AS_Q = AS_Q.unsqueeze(2)
        fmask = fmask.unsqueeze(1)
        scale = math.sqrt(DW_K.size(-1)*scale_factor)
        if 'AS_TW' in self.args.multi_disen_att_type:
            
            AS_TW_att = AS_TW_att/scale
            AS_TW_att = AS_TW_att.squeeze(2)
            if self.args.softmax_first:
                AS_TW_att = mask_logits(AS_TW_att, fmask)
                AS_TW_att = torch.softmax(AS_TW_att, dim=-1)
            att_score = [AS_TW_att]
        else:
            att_score = []
        
        if 'AS_Wi' in self.args.multi_disen_att_type:
            # W_KT = W_K * AS_TW_att.transpose(-1, -2).unsqueeze(-1)     # W_K  4*5*57*128
            W_K = W_K * fmask.unsqueeze(-1)
            if self.args.wi_use_ntn:
                ntn_g = self.get_ntn_ft_for_edge(AS_Q, W_K, fmask)
                AS_Wi_att = self.calcu_gate(ntn_g, self.gate1_ntn, self.gate2_ntn)
            else:
                AS_Wi_att = torch.matmul(AS_Q, W_K.transpose(-1, -2))
            AS_Wi_att = AS_Wi_att/scale
            AS_Wi_att = AS_Wi_att.squeeze(2)
            if self.args.softmax_first:
                AS_Wi_att = mask_logits(AS_Wi_att, fmask)
                AS_Wi_att = torch.softmax(AS_Wi_att, dim=-1)   #4*5*57
            att_score += [AS_Wi_att]
        
        if 'AS_DW' in self.args.multi_disen_att_type:
            DW_K = DW_K * fmask.unsqueeze(-1)
            if self.args.implicit_edge == True:
                ntn_f = self.get_ntn_ft_for_edge(AS_Q, W_K, fmask)
                ntn_f = ntn_f * fmask.unsqueeze(-1)
                DW_K = self.trans(torch.cat([DW_K, ntn_f], dim=-1))
            AS_DW_att = torch.matmul(AS_Q, DW_K.transpose(-1, -2))
            AS_DW_att = AS_DW_att/scale
            AS_DW_att = AS_DW_att.squeeze(2)
            if self.args.softmax_first:   
                AS_DW_att = mask_logits(AS_DW_att, fmask)
                AS_DW_att = torch.softmax(AS_DW_att, dim=-1)
            att_score += [AS_DW_att]

        att_a = self.attcombination(att_score)

        if not self.args.softmax_first:
            att_a = mask_logits(att_a, fmask)
            att_a = torch.softmax(att_a, dim=-1)

        W_V = self.rel_transform(W_V * fmask.unsqueeze(-1), T_V * fmask.unsqueeze(-1))
        
        att_z = torch.matmul(att_a.unsqueeze(-2), W_V * fmask.unsqueeze(-1))
        att_z = att_z.squeeze(2)
        att_z = self.dropout(att_z)        

        return att_z

    def get_ntn_ft_for_edge(self, AS_Q, W_K, fmask):
        #################NTN#######################
        # utilize ntn to judge the relations between two node
        # linear production
        # AS_Q = AS_Q.unsqueeze(2)
        B, A, num_node, hdim = W_K.shape
        AS_Q_R = torch.repeat_interleave(AS_Q, num_node, dim=-2)     # 8*5*57*128
        AS_Q_R = AS_Q_R * fmask.unsqueeze(-1)
        AS_Q_R_W_K = torch.cat([AS_Q_R, W_K], dim=-1)
        lp = self.W1(AS_Q_R_W_K)

        # Tensor production
        AS_Q_R = AS_Q_R.view(-1, num_node, hdim)
        T = self.T1.view(hdim, -1)
        AR_T = torch.matmul(AS_Q_R, T)   # B*A, num_node, hdim*hdim
        AR_T = AR_T.view(-1, hdim, hdim)   #B*A*num_node, hdim, hdim
        W_K_R = W_K.view(-1, hdim).unsqueeze(-1)   #B*A*num_node, hdim， 1
        tp = torch.matmul(AR_T, W_K_R).squeeze(-1).view(B, A, num_node, hdim)   # B*A*num_node, hdim, 1
        ntn_f = tp + lp

        return ntn_f


    def forward(self, feature, dep_feature, aspect_feature, all_type_feature, fmask): #type_feature, shortest_path_feature,
        '''
            DA is the dep tag of aspect
            DW is the dep tag of words
            att_type = ['AS_Wi', 'AS_DW', 'DA_W', 'DA_DW']

        '''
        ###########trans dep feature#######

        DW_K = self.key(dep_feature) if self.args.deps_share_weight_with_nodes else self.key_dep(dep_feature)   # combined_dep_tags + SNP

        ###########trans node feature###########
        AS_Q = self.query(aspect_feature)
        W_K = self.key(feature)  #aspects feature + feature
        W_V = self.value(feature)
        
        #############trans type features###############
        T_K = self.key(all_type_feature) if self.args.deps_share_weight_with_nodes else self.key_type(all_type_feature)
        T_V = self.value(all_type_feature) if self.args.deps_share_weight_with_nodes else self.value_type(all_type_feature)
        ##################calcu attention######################
        # scale_factor = len(self.args.multi_disen_att_type)  #4
        scale_factor = 1
        att_AS_TW_att = self.type_att(AS_Q, T_K, scale_factor_att=scale_factor)
        out = self.att_head(AS_Q, W_K, W_V, DW_K, T_V, att_AS_TW_att, scale_factor, fmask)
      
        return out



