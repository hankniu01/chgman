from math import e
from torch.nn.utils.rnn import pad_sequence
import argparse
import json
import linecache
import logging
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from copy import copy, deepcopy

import nltk
import numpy as np
import json
import torch
from lxml import etree
from nltk import word_tokenize
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import networkx as nx


logger = logging.getLogger(__name__)

def not_split_load_dataseta_and_vocabs(args):
    train, test = get_dataset(args.dataset_name.split('_')[0])

    # Our model takes unrolled data, currently we don't consider the MAMS cases(future experiments)
    train_all_rolled, train_all_unrolled, train_mixed_rolled, _ = get_rolled_and_unrolled_data(train, args)
    test_all_rolled, test_all_unrolled, test_mixed_rolled, _ = get_rolled_and_unrolled_data(test, args)

    logger.info('****** After unrolling ******')
    logger.info('Train set size: %s', len(train_all_unrolled))
    logger.info('Test set size: %s,', len(test_all_unrolled))

    # Build word vocabulary(part of speech, dep_tag) and save pickles.
    word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab = load_and_cache_vocabs(
        train_all_unrolled+test_all_unrolled, args)

    train_no_split = No_Split_Data(train_all_rolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    overall_train_data, dep_tag_vocab = train_no_split.output()

    overall_train_dataset = ASBA_Depparsed_Dataset_For_Overall(
        overall_train_data, args)

    test_no_split = No_Split_Data(test_all_rolled, args, word_vocab, dep_tag_vocab, pos_tag_vocab)
    overall_test_data, dep_tag_vocab = test_no_split.output()

    overall_test_dataset = ASBA_Depparsed_Dataset_For_Overall(
        overall_test_data, args)

    return overall_train_dataset, overall_test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab


def read_sentence_depparsed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data


def get_dataset(dataset_name):
    '''
    Already preprocess the data and now they are in json format.(only for semeval14)
    Retrieve train and test set
    With a list of dict:
    e.g. {"sentence": "Boot time is super fast, around anywhere from 35 seconds to 1 minute.",
    "tokens": ["Boot", "time", "is", "super", "fast", ",", "around", "anywhere", "from", "35", "seconds", "to", "1", "minute", "."],
    "tags": ["NNP", "NN", "VBZ", "RB", "RB", ",", "RB", "RB", "IN", "CD", "NNS", "IN", "CD", "NN", "."],
    "predicted_dependencies": ["nn", "nsubj", "root", "advmod", "advmod", "punct", "advmod", "advmod", "prep", "num", "pobj", "prep", "num", "pobj", "punct"],
    "predicted_heads": [2, 3, 0, 5, 3, 5, 8, 5, 8, 11, 9, 9, 14, 12, 3],
    "dependencies": [["nn", 2, 1], ["nsubj", 3, 2], ["root", 0, 3], ["advmod", 5, 4], ["advmod", 3, 5], ["punct", 5, 6], ["advmod", 8, 7], ["advmod", 5, 8],
                    ["prep", 8, 9], ["num", 11, 10], ["pobj", 9, 11], ["prep", 9, 12], ["num", 14, 13], ["pobj", 12, 14], ["punct", 3, 15]],
    "aspect_sentiment": [["Boot time", "positive"]], "from_to": [[0, 2]]}
    '''
    rest_train = './data/semeval14/Restaurants_Train_v2_biaffine_depparsed_with_energy.json'
    rest_test = './data/semeval14/Restaurants_Test_Gold_biaffine_depparsed_with_energy.json'

    laptop_train = './data/semeval14/Laptop_Train_v2_biaffine_depparsed.json'
    laptop_test = './data/semeval14/Laptops_Test_Gold_biaffine_depparsed.json'

    MAMS_train = './data/MAMS-ATSA/raw/train_biaffine_depparsed.json'
    MAMS_test = './data/MAMS-ATSA/raw/test_biaffine_depparsed.json'

    ds_train = {'rest': rest_train,
                'laptop': laptop_train, 'mams': MAMS_train}
    ds_test = {'rest': rest_test,
               'laptop': laptop_test, 'mams': MAMS_test}

    train = list(read_sentence_depparsed(ds_train[dataset_name]))
    logger.info('# Read %s Train set: %d', dataset_name, len(train))

    test = list(read_sentence_depparsed(ds_test[dataset_name]))
    logger.info("# Read %s Test set: %d", dataset_name, len(test))
    return train, test


def reshape_dependency_tree_new(as_start, as_end, dependencies, multi_hop=False, add_non_connect=False, tokens=None, max_hop = 5):
    '''
    Adding multi hops
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.
    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''
    dep_tag = []
    dep_idx = []
    dep_dir = []
    # 1 hop

    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            dep_tag_temp = deepcopy(dep_tag)
            for i, d_tag in zip(dep_idx_temp, dep_tag_temp):
                d_tag_ls = d_tag.split(':')
                dt = d_tag if len(d_tag_ls) == 1 else d_tag[:-2]
                for dep in dependencies:
                    if i == dep[1] - 1:
                        # not root, not aspect
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                # dep_tag.append('ncon_'+str(current_hop))
                                dep_tag.append(dt + ':' + dep[0] + ':' +str(current_hop))
                                dep_dir.append(1)
                            else:
                                # dep_tag.append('<pad>')
                                dep_tag.append(dt + ':' + '<pad>' + ':' + str(current_hop))
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                # dep_tag.append('ncon_'+str(current_hop))
                                dep_tag.append(dt + ':' + dep[0] + ':' + str(current_hop))
                                dep_dir.append(2)
                            else:
                                # dep_tag.append('<pad>')
                                dep_tag.append(dt + ':' + '<pad>' + ':' + str(current_hop))
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir


def rgat_raw_reshape_dependency_tree_new(as_start, as_end, dependencies, multi_hop=False, add_non_connect=False, tokens=None, max_hop = 5):
    '''
    Adding multi hops
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.
    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''
    dep_tag = []
    dep_idx = []
    dep_dir = []
    # 1 hop

    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            for i in dep_idx_temp:
                for dep in dependencies:
                    if i == dep[1] - 1:
                        # not root, not aspect
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append('ncon_'+str(current_hop))
                                dep_dir.append(2)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir


def reshape_dependency_tree(as_start, as_end, dependencies, add_2hop=False, add_non_connect=False, tokens=None):
    '''
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.

    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    '''
    dep_tag = []
    dep_idx = []
    dep_dir = []
    # 1 hop

    for i in range(as_start, as_end):
        for dep in dependencies:
            if i == dep[1] - 1:
                # not root, not aspect
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    # 2 hop
    if add_2hop:
        dep_idx_cp = dep_idx
        for i in dep_idx_cp:
            for dep in dependencies:
                # connect to i, not a punct
                if i == dep[1] - 1 and str(dep[0]) != 'punct':
                    # not root, not aspect
                    if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0:
                        if dep[2]-1 not in dep_idx:
                            dep_tag.append(dep[0])
                            dep_idx.append(dep[2] - 1)
                # connect to i, not a punct
                elif i == dep[2] - 1 and str(dep[0]) != 'punct':
                    # not root, not aspect
                    if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0:
                        if dep[1]-1 not in dep_idx:
                            dep_tag.append(dep[0])
                            dep_idx.append(dep[1] - 1)
    if add_non_connect:
        for idx, token in enumerate(tokens):
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x:x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir

def get_shortest_path_nodes(dependencies, aspect_index):
    SPNs = []
    edges = [(dep[1], dep[2]) for dep in dependencies]
    G = nx.Graph(edges)
    num_node = G.number_of_nodes() - 1
    length = len(aspect_index)
    # for idx in range(length):
    #     for k in range(1, length-idx):
    #         SPNs += [[aspect_index[idx][0], aspect_index[idx+k][0], nx.shortest_path(G, source=min(aspect_index[idx][1], num_node), target=min(aspect_index[idx+k][1], num_node))]]
    for idx in range(length):
        for k in range(length):
            if k == idx:
                SPNs += [[aspect_index[idx][0], aspect_index[k][0], [0]]]
            else:
                SPNs += [[aspect_index[idx][0], aspect_index[k][0], nx.shortest_path(G, source=min(aspect_index[idx][1], num_node), target=min(aspect_index[k][1], num_node))]]
    re_SPNs = []
    for spn in SPNs:
        if spn[2] != [0]:
            spnslist = [val + 1 for val in spn[2]]
            re_SPNs += [[spn[0], spn[1], spnslist]]
        else:
            re_SPNs += [spn]
    return re_SPNs

def get_rolled_and_unrolled_data(input_data, args):
    '''
    In input_data, each sentence could have multiple aspects with different sentiments.
    Our method treats each sentence with one aspect at a time, so even for
    multi-aspect-multi-sentiment sentences, we will unroll them to single aspect sentence.

    Perform reshape_dependency_tree to each sentence with aspect

    return:
        all_rolled:
                a list of dict
                    {sentence, tokens, pos_tags, pos_class, aspects(list of aspects), sentiments(list of sentiments)
                    froms, tos, dep_tags, dep_index, dependencies}
        all_unrolled:
                unrolled, with aspect(single), sentiment(single) and so on...
        mixed_rolled:
                Multiple aspects and multiple sentiments, ROLLED.
        mixed_unrolled:
                Multiple aspects and multiple sentiments, UNROLLED.
    '''
    # A hand-picked set of part of speech tags that we see contributes to ABSA.
    opinionated_tags = ['JJ', 'JJR', 'JJS', 'RB', 'RBR',
                        'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    all_rolled = []
    all_unrolled = []
    mixed_rolled = []
    mixed_unrolled = []

    unrolled = []
    mixed = []
    unrolled_ours = []
    mixed_ours = []

    # Make sure the tree is successfully built.
    zero_dep_counter = 0

    # Sentiment counters
    total_counter = defaultdict(int)
    mixed_counter = defaultdict(int)
    sentiments_lookup = {'negative': 0, 'positive': 1, 'neutral': 2}
    relation_lookup = {'equal': 0, 'reverse': 1, 'non-reverse': 2}

    logger.info('*** Start processing data(unrolling and reshaping) ***')
    tree_samples = []
    # for seeking 'but' examples
    for e in input_data:
        e['tokens'] = [x.lower() for x in e['tokens']]
        aspects = []
        sentiments = []
        froms = []
        tos = []
        dep_tags = []
        dep_index = []
        dep_dirs = []
        aspect_relation = []
        aspect_index = []

        # Classify based on POS-tags

        pos_class = e['tags']

        # Iterate through aspects in a sentence and reshape the dependency tree.
        for i in range(len(e['aspect_sentiment'])):
            aspect = e['aspect_sentiment'][i][0].lower()
            aspect_ = aspect
            # We would tokenize the aspect while at it.
            if '-' in aspect:
                # aspect = ['wine', '-', 'by', 'the', '-', 'glass']
                aspect = args.tokenizer.tokenize(aspect)
                # aspect = word_tokenize(aspect)
            elif aspect == 'ambiance':
                aspect = 'ambiance-'
            else:
                aspect = word_tokenize(aspect)
            sentiment = sentiments_lookup[e['aspect_sentiment'][i][1]]
            frm = e['from_to'][i][0]
            to = e['from_to'][i][1]
            if i <= len(e['aspect_sentiment'])-1:
                for k in range(1, len(e['aspect_sentiment'])-i):
                    aspect_1 = e['aspect_sentiment'][i+k][0].lower()
                    if e['aspect_sentiment'][i][1] == e['aspect_sentiment'][i+k][1]:   #relation_lookup = {'equal': 0, 'reverse': 1, 'non-reverse': 2}
                        aspect_relation += [(aspect_, aspect_1, relation_lookup['equal'])]
                    if e['aspect_sentiment'][i][1] != e['aspect_sentiment'][i+k][1] and 'neutral' not in [e['aspect_sentiment'][i][1], e['aspect_sentiment'][i+k][1]]:   #relation_lookup = {'equal': 0, 'reverse': 1, 'non-reverse': 2}
                        aspect_relation += [(aspect_, aspect_1, relation_lookup['reverse'])]
                    if e['aspect_sentiment'][i][1] != e['aspect_sentiment'][i+k][1] and 'neutral' in [e['aspect_sentiment'][i][1], e['aspect_sentiment'][i+k][1]]:   #relation_lookup = {'equal': 0, 'reverse': 1, 'non-reverse': 2}
                        aspect_relation += [(aspect_, aspect_1, relation_lookup['non-reverse'])]

            aspects.append(aspect)
            sentiments.append(sentiment)
            froms.append(frm)
            tos.append(to)

            ############get aspect_index############
            # if aspect not in aspect_index.keys():
            aspect_index += [[aspect_, to]]
            
            # Center on the aspect.
            if args.dep_tag_type == 'composed':
                dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(frm, to, e['dependencies'],
                                                           multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)
            elif args.dep_tag_type == 'n_conn':
                dep_tag, dep_idx, dep_dir = rgat_raw_reshape_dependency_tree_new(frm, to, e['dependencies'],
                                                        multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)

            # Because of tokenizer differences, aspect opsitions are off, so we find the index and try again.
            if len(dep_tag) == 0:
                zero_dep_counter += 1
                as_sent = e['aspect_sentiment'][i][0].split()
                as_start = e['tokens'].index(as_sent[0])
                # print(e['tokens'], e['aspect_sentiment'], e['dependencies'],as_sent[0])
                as_end = e['tokens'].index(
                    as_sent[-1]) if len(as_sent) > 1 else as_start + 1
                print("Debugging: as_start as_end ", as_start, as_end)
                dep_tag, dep_idx, dep_dir = reshape_dependency_tree_new(as_start, as_end, e['dependencies'],
                                                           multi_hop=args.multi_hop, add_non_connect=args.add_non_connect, tokens=e['tokens'], max_hop=args.max_hop)
                if len(dep_tag) == 0:  # for debugging
                    print("Debugging: zero_dep",
                          e['aspect_sentiment'][i][0], e['tokens'])
                    print("Debugging: ". e['dependencies'])
                else:
                    zero_dep_counter -= 1

            dep_tags.append(dep_tag)
            dep_index.append(dep_idx)
            dep_dirs.append(dep_dir)

            total_counter[e['aspect_sentiment'][i][1]] += 1

            # Unrolling
            all_unrolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect, 'sentiment': sentiment,
                    'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir':dep_dir,'dependencies': e['dependencies']})

            #temp
            if 'pork' in e['tokens'] and 'fried' in e['tokens']:
                print({'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspect, 'sentiment': sentiment,
                    'predicted_dependencies': e['predicted_dependencies'], 'predicted_heads': e['predicted_heads'],
                 'from': frm, 'to': to, 'dep_tag': dep_tag, 'dep_idx': dep_idx, 'dep_dir':dep_dir,'dependencies': e['dependencies']})

        # All sentences with multiple aspects and sentiments rolled.
        #### To get shortest path node ids for every two aspect pairs #####
        if len(aspect_index) > 1:
            SPNs = get_shortest_path_nodes(e['dependencies'], aspect_index)
        else:
            SPNs = []

        all_rolled.append(
            {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
             'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_index': dep_index, 'dependencies': e['dependencies'],
            'predicted_dependencies': e['predicted_dependencies'], 'dep_dir':dep_dir, 'aspect_relation':aspect_relation, 'SPNs': SPNs})

        # Ignore sentences with single aspect or no aspect
        if len(e['aspect_sentiment']) and len(set(map(lambda x: x[1], e['aspect_sentiment']))) > 1:
            mixed_rolled.append(
                {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspects': aspects, 'sentiments': sentiments,
                 'from': froms, 'to': tos, 'dep_tags': dep_tags, 'dep_index': dep_index, 'dependencies': e['dependencies']})

            # Unrolling
            for i, as_sent in enumerate(e['aspect_sentiment']):
                mixed_counter[as_sent[1]] += 1
                mixed_unrolled.append(
                    {'sentence': e['tokens'], 'tags': e['tags'], 'pos_class': pos_class, 'aspect': aspects[i], 'sentiment': sentiments[i],
                     'from': froms[i], 'to': tos[i], 'dep_tag': dep_tags[i], 'dep_idx': dep_index[i], 'dependencies': e['dependencies']})


    logger.info('Total sentiment counter: %s', total_counter)
    logger.info('Multi-Aspect-Multi-Sentiment counter: %s', mixed_counter)

    return all_rolled, all_unrolled, mixed_rolled, mixed_unrolled


def load_and_cache_vocabs(data, args):
    '''
    Build vocabulary of words, part of speech tags, dependency tags and cache them.
    Load glove embedding if needed.
    '''
    pkls_path = os.path.join(args.output_dir, 'pkls')
    if not os.path.exists(pkls_path):
        os.makedirs(pkls_path)

    # Build or load word vocab and glove embeddings.
    # Elmo and bert have it's own vocab and embeddings.
    if args.embedding_type == 'glove':
        cached_word_vocab_file = os.path.join(
            pkls_path, 'cached_{}_{}_word_vocab.pkl'.format(args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vocab_file):
            logger.info('Loading word vocab from %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'rb') as f:
                word_vocab = pickle.load(f)
        else:
            logger.info('Creating word vocab from dataset %s',
                        args.dataset_name)
            word_vocab = build_text_vocab(data)
            logger.info('Word vocab size: %s', word_vocab['len'])
            logging.info('Saving word vocab to %s', cached_word_vocab_file)
            with open(cached_word_vocab_file, 'wb') as f:
                pickle.dump(word_vocab, f, -1)

        cached_word_vecs_file = os.path.join(pkls_path, 'cached_{}_{}_word_vecs.pkl'.format(
            args.dataset_name, args.embedding_type))
        if os.path.exists(cached_word_vecs_file):
            logger.info('Loading word vecs from %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'rb') as f:
                word_vecs = pickle.load(f)
        else:
            logger.info('Creating word vecs from %s', args.glove_dir)
            word_vecs = load_glove_embedding(
                word_vocab['itos'], args.glove_dir, 0.25, args.embedding_dim)
            logger.info('Saving word vecs to %s', cached_word_vecs_file)
            with open(cached_word_vecs_file, 'wb') as f:
                pickle.dump(word_vecs, f, -1)
    else:
        word_vocab = None
        word_vecs = None

    # Build vocab of dependency tags
    cached_dep_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_dep_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_dep_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'rb') as f:
            dep_tag_vocab = pickle.load(f)
        ########add root to default dep_tag_vocab dict for Disentangle GAT##############
        dep_tag_vocab['stoi']['root'] = 46
        dep_tag_vocab['itos'] += ['root']
    else:
        logger.info('Creating vocab of dependency tags.')
        dep_tag_vocab = build_dep_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    dep_tag_vocab['len'], cached_dep_tag_vocab_file)
        with open(cached_dep_tag_vocab_file, 'wb') as f:
            pickle.dump(dep_tag_vocab, f, -1)

    # Build vocab of part of speech tags.
    cached_pos_tag_vocab_file = os.path.join(
        pkls_path, 'cached_{}_pos_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_pos_tag_vocab_file):
        logger.info('Loading vocab of dependency tags from %s',
                    cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'rb') as f:
            pos_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of dependency tags.')
        pos_tag_vocab = build_pos_tag_vocab(data, min_freq=0)
        logger.info('Saving dependency tags  vocab, size: %s, to file %s',
                    pos_tag_vocab['len'], cached_pos_tag_vocab_file)
        with open(cached_pos_tag_vocab_file, 'wb') as f:
            pickle.dump(pos_tag_vocab, f, -1)

    return word_vecs, word_vocab, dep_tag_vocab, pos_tag_vocab


def load_glove_embedding(word_list, glove_dir, uniform_scale, dimension_size):
    glove_words = []
    with open(os.path.join('/home/niuhao/Document_dating/glove/word2vec.840B.300d.txt'), 'r') as fopen:
        for line in fopen:
            glove_words.append(line.strip().split(' ')[0])
    word2offset = {w: i for i, w in enumerate(glove_words)}
    word_vectors = []
    for word in word_list:
        if word in word2offset:
            line = linecache.getline(os.path.join(
                '/home/niuhao/Document_dating/glove/word2vec.840B.300d.txt'), word2offset[word]+1)
            assert(word == line[:line.find(' ')].strip())
            word_vectors.append(np.fromstring(
                line[line.find(' '):].strip(), sep=' ', dtype=np.float32))
        elif word == '<pad>':
            word_vectors.append(np.zeros(dimension_size, dtype=np.float32))
        else:
            word_vectors.append(
                np.random.uniform(-uniform_scale, uniform_scale, dimension_size))
    return word_vectors


def _default_unk_index():
    return 1


def build_text_vocab(data, vocab_size=100000, min_freq=2):
    counter = Counter()
    for d in data:
        s = d['sentence']
        counter.update(s)

    itos = ['[PAD]', '[UNK]']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}


def build_pos_tag_vocab(data, vocab_size=1000, min_freq=1):
    """
    Part of speech tags vocab.
    """
    counter = Counter()
    for d in data:
        tags = d['tags']
        counter.update(tags)

    itos = ['<pad>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}



def build_dep_tag_vocab(data, vocab_size=1000, min_freq=0):
    counter = Counter()
    for d in data:
        tags = d['dep_tag']
        counter.update(tags)

    itos = ['<pad>', '<unk>']
    min_freq = max(min_freq, 1)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word == '<pad>':
            continue
        itos.append(word)
    # stoi is simply a reverse dict for itos
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

class No_Split_Data(object):
    def __init__(self, data, args, word_vocab, dep_tag_vocab, pos_tag_vocab):
        self.data = data
        self.args = args
        self.word_vocab = word_vocab
        self.dep_tag_vocab = dep_tag_vocab
        self.pos_tag_vocab = pos_tag_vocab
        self.overall_data = []
        self.overall_idx = 0
        for id in range(1,10):
            self.dep_tag_vocab['stoi'][str(id)] = 46+id
    
    def output(self):
        self.convert_features()
        return self.overall_data, self.dep_tag_vocab
    
    def convert_features(self):
        '''
        Convert sentence, aspects, pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.data)):
            if self.args.embedding_type == 'glove':
                self.data[i]['sentence_ids'] = [self.word_vocab['stoi'][w]
                                                for w in self.data[i]['sentence']]
                self.data[i]['aspect_ids'] = [self.word_vocab['stoi'][w]
                                              for w in self.data[i]['aspects']]
            elif self.args.embedding_type == 'elmo':
                self.data[i]['sentence_ids'] = self.data[i]['sentence']
                self.data[i]['aspect_ids'] = self.data[i]['aspects']
            else:  # self.args.embedding_type == 'bert'
                self.overall_convert_features_bert(i)
    
    def overall_convert_features_bert(self, i):

        cls_token = "[CLS]"
        sep_token = "[SEP]"

        self.overall_data += [self.data[i]]
        tokens = []
        word_indexer = []
        aspect_tokens = []
        aspect_indexer = []

        for word in self.data[i]['sentence']:
            word_tokens = self.args.tokenizer.tokenize(word)
            token_idx = len(tokens)
            tokens.extend(word_tokens)
            # word_indexer is for indexing after bert, feature back to the length of original length.
            word_indexer.append(token_idx)
        
        for word_list in self.data[i]['aspects']:
            as_tok, as_ind = [], []
            for word in word_list:
                word_aspect_tokens = self.args.tokenizer.tokenize(word)
                token_idx = len(aspect_tokens)
                as_tok.extend(word_aspect_tokens)
                as_ind.append(token_idx)
            aspect_tokens += [as_tok]
            aspect_indexer += [as_ind]
        
        aspect_tokens_list = []
        aspect_indexer_list = []
        tokens = [cls_token] + tokens + [sep_token]
        for as_tok, as_ind in zip(aspect_tokens, aspect_indexer):
            aspect_tokens_list += [[cls_token] + as_tok + [sep_token]]
            aspect_indexer_list += [[i+1 for i in as_ind]]
        word_indexer = [i+1 for i in word_indexer]

        input_aspect_ids_list = []
        input_ids = self.args.tokenizer.convert_tokens_to_ids(tokens)
        input_aspect_ids_list += [self.args.tokenizer.convert_tokens_to_ids(
            aspect_tokens) for aspect_tokens in aspect_tokens_list]

        # check len of word_indexer equals to len of sentence.
        assert len(word_indexer) == len(self.data[i]['sentence'])
        assert len(aspect_indexer) == len(self.data[i]['aspects'])

        if self.args.pure_bert:
            input_cat_ids = input_ids + input_aspect_ids[1:]
            segment_ids = [0] * len(input_ids) + [1] * len(input_aspect_ids[1:])

            self.overall_data[self.overall_idx]['input_cat_ids'] = input_cat_ids
            self.overall_data[self.overall_idx]['segment_ids'] = segment_ids
        else:           
            self.overall_data[self.overall_idx]['is_multi_aspect'] = 'true'
            self.overall_data[self.overall_idx]['input_cat_ids'] = [input_ids + input_aspect_ids[1:] for input_aspect_ids in input_aspect_ids_list]
            self.overall_data[self.overall_idx]['segment_ids'] = [[0] * len(input_ids) + [1] * len(input_aspect_ids[1:]) for input_aspect_ids in input_aspect_ids_list]
            self.overall_data[self.overall_idx]['input_ids'] = input_ids
            self.overall_data[self.overall_idx]['word_indexer'] = word_indexer
            self.overall_data[self.overall_idx]['input_aspect_ids'] = input_aspect_ids_list
            self.overall_data[self.overall_idx]['aspect_indexer'] = aspect_indexer
            self.overall_data[self.overall_idx]['SPN'] = [spns[-1] for spns in self.overall_data[self.overall_idx]['SPNs']] if self.overall_data[self.overall_idx]['SPNs'] != [] else []

        self.overall_data[self.overall_idx]['text_len'] = len(self.overall_data[self.overall_idx]['sentence'])
        self.overall_data[self.overall_idx]['aspect_position'] = [0] * self.overall_data[self.overall_idx]['text_len']
        
        if self.args.dep_embed == 'bert':
            self.overall_data[self.overall_idx]['dep_tag_ids'] = [self.args.tokenizer(ws, padding='max_length', max_length=30)['input_ids'] for ws in self.overall_data[self.overall_idx]['dep_tags']]
            self.overall_data[self.overall_idx]['aspect_dep_tag_ids'] = [self.args.tokenizer('root')['input_ids']]
        elif self.args.dep_embed == 'combine':
            self.overall_data[self.overall_idx]['dep_tag_ids'] = []
            for ws in self.overall_data[self.overall_idx]['dep_tags']:
                dep_tag_ids_list = []
                for w in ws:
                    if w in self.dep_tag_vocab['stoi'].keys():
                        dep_tag_ids_list += [torch.tensor([self.dep_tag_vocab['stoi'][w]])]
                    else:
                        dep_tag_ids_list += [torch.tensor([self.dep_tag_vocab['stoi'][w_] for w_ in w.split(':')])]
                self.overall_data[self.overall_idx]['dep_tag_ids'] += [dep_tag_ids_list]

            self.overall_data[self.overall_idx]['aspect_dep_tag_ids'] = [self.dep_tag_vocab['stoi']['root']]
        elif self.args.dep_embed == 'random':
            self.overall_data[self.overall_idx]['dep_tag_ids'] = [[self.dep_tag_vocab['stoi'][w] for w in ws]
                                        for ws in self.overall_data[self.overall_idx]['dep_tags']]
            self.overall_data[self.overall_idx]['aspect_dep_tag_ids'] = [self.dep_tag_vocab['stoi']['root']]

        self.overall_data[self.overall_idx]['dep_dir_ids'] = [idx
                                        for idx in self.overall_data[self.overall_idx]['dep_dir']]
        self.overall_data[self.overall_idx]['pos_class'] = [self.pos_tag_vocab['stoi'][w]
                                            for w in self.overall_data[self.overall_idx]['tags']]
        self.overall_data[self.overall_idx]['aspect_len'] = len(self.overall_data[self.overall_idx]['aspects'])

        self.overall_data[self.overall_idx]['dep_rel_ids'] = [self.dep_tag_vocab['stoi'][r]
                                        for r in self.overall_data[self.overall_idx]['predicted_dependencies']]

        try:  # find the index of aspect in sentence
            for j in range(self.overall_data[self.overall_idx]['from'], self.overall_data[self.overall_idx]['to']):
                self.overall_data[self.overall_idx]['aspect_position'][j] = 1
        except:
            pass

        try:                                         
            for termlist in self.overall_data[self.overall_idx]['aspects']:
                for term in termlist:
                    self.overall_data[self.overall_idx]['aspect_position'][self.overall_data[self.overall_idx]
                                                ['sentence'].index(term)] = 1
        except:
            pass

        self.overall_idx += 1



class ASBA_Depparsed_Dataset_For_Overall(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        items_tensor, items_tensor = [], []
        items = e['pos_class'], e['text_len'], e['aspect_len'], e['sentiments'],\
            e['dep_rel_ids'], e['dep_dir_ids'], e['aspect_dep_tag_ids']  #e['aspect_position'],
        if self.args.embedding_type == 'glove':
            non_bert_items = e['sentence_ids'], e['aspect_ids']
            items_tensor = non_bert_items + items
            items_tensor = tuple(torch.tensor(t) for t in items_tensor)
        elif self.args.embedding_type == 'elmo':
            items_tensor = e['sentence_ids'], e['aspect_ids']
            items_tensor += tuple(torch.tensor(t) for t in items)
        else:  # bert
            if self.args.pure_bert:
                bert_items = e['input_cat_ids'], e['segment_ids']
                items_tensor = tuple(torch.tensor(t) for t in bert_items)
                items_tensor += tuple(torch.tensor(t) for t in items)
            else:
                bert_items = e['input_ids'], e['word_indexer']
                bert_items_multi = e['input_aspect_ids'], e['aspect_indexer'], e['input_cat_ids'], e['segment_ids'], e['dep_tag_ids'], e['SPN']
                # segment_id
                items_tensor = [torch.tensor(t) for t in bert_items]
                length_dict = {}
                bert_items_multi_list = ['input_aspect_ids', 'aspect_indexer', 'input_cat_ids', 'segment_ids', 'dep_tag_ids', 'SPN']
                for ts, ts_type in zip(bert_items_multi, bert_items_multi_list):
                    tt = []
                    length_dict[ts_type] = []
                    if ts_type == 'dep_tag_ids':
                        length_dict[ts_type+'_outer'] = []
                    for t in ts:
                        if ts_type == 'dep_tag_ids':
                            tt += [pad_sequence(t, batch_first=True, padding_value=0)]
                            if self.args.dep_embed == 'bert':
                                length_dict[ts_type] += [len(t)]
                            elif self.args.dep_embed == 'combine':
                                length_dict[ts_type] += [tt[-1].shape[-1]]
                                length_dict[ts_type+'_outer'] += [tt[-1].shape[-2]]
                        else:                                
                            tt += [torch.tensor(t)]
                            length_dict[ts_type] += [len(t)]
                    if ts_type != 'dep_tag_ids' and len(e['sentiments']) > 1:
                        tt = pad_sequence(tt, batch_first=True, padding_value=0)
                    elif ts_type != 'SPN' and ts_type != 'dep_tag_ids':
                        tt = tt[0].unsqueeze(0)
                    if ts_type == 'SPN':
                        num_asp = len(e['sentiments'])
                        if num_asp > 1:
                            SPN_ = torch.zeros(num_asp, num_asp, items_tensor[4].shape[-1])   #[num_asp, num_asp, num_word]
                            SPN = tt.view(num_asp, -1, tt.shape[-1])  #[num_asp, num_asp, num_spn]
                            for m, spn in enumerate(SPN):
                                for n, sp in enumerate(spn): #[num_asp, num_spn]
                                    if sum(sp.tolist()) != 0:
                                        SPN_[m][n][sp] = 1
                            tt = SPN_.view(-1, SPN_.shape[-1])
                        else:
                            tt = torch.tensor([0]).unsqueeze(0)
                    assert tt != [], 'wrong tt'
                    items_tensor += [tt]
                if e['aspect_relation'] != []:
                    # items_tensor += [[torch.tensor(asr[-1]) for asr in e['aspect_relation']]]
                    items_tensor += [torch.tensor([asr[-1] for asr in e['aspect_relation']])]
                else:
                    items_tensor += [torch.tensor([0])]
                items_tensor += [torch.tensor(t) for t in items]
                items_tensor += [length_dict]
        return items_tensor


def my_collate_bert_for_overall(batch):
    '''
    Pad sentence and aspect in a batch.
    Sort the sentences based on length.
    Turn all into tensors.

    Process bert feature

    bert_items_multi = e['input_aspect_ids'], e['aspect_indexer'], e['input_cat_ids'], e['segment_ids'], e['dep_tag_ids']
    '''
    # input_ids, word_indexer, input_aspect_ids, aspect_indexer, input_cat_ids, segment_ids, dep_tag_ids, SPN, aspect_relation, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_dir_ids, aspect_dep_tag_ids, length_dict = zip(*batch)
    input_ids, word_indexer, input_aspect_ids, aspect_indexer, input_cat_ids, segment_ids, dep_tag_ids, SPN, aspect_relation, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_dir_ids, aspect_dep_tag_ids, length_dict = zip(*batch)
    text_len = torch.tensor(text_len)
    aspect_len = torch.tensor(aspect_len)
    # sentiment = torch.tensor(sentiment)

    #Pad multi sample
    max_len_dict = {}
    for length_item in length_dict:
        for (key, val) in length_item.items():
            if val != []:
                if key not in max_len_dict.keys() or max(val) > max_len_dict[key]:
                    max_len_dict[key] = max(val)
    new_input_aspect_ids, new_aspect_indexer, new_input_cat_ids, new_segment_ids, new_dep_tag_ids, new_SPN = [],[],[],[],[],[]
    for input_aspect_ids_item, aspect_indexer_item, input_cat_ids_item, segment_ids_item, dep_tag_ids_item, spn_item in zip(input_aspect_ids, aspect_indexer, input_cat_ids, segment_ids, dep_tag_ids, SPN):
        new_input_aspect_ids += [F.pad(input_aspect_ids_item, (0, max_len_dict['input_aspect_ids']-input_aspect_ids_item.shape[1]), mode='constant', value=0)]
        new_aspect_indexer += [F.pad(aspect_indexer_item, (0, max_len_dict['aspect_indexer']-aspect_indexer_item.shape[1]), mode='constant', value=1)]
        new_input_cat_ids += [F.pad(input_cat_ids_item, (0, max_len_dict['input_cat_ids']-input_cat_ids_item.shape[1]), mode='constant', value=0)]
        new_segment_ids += [F.pad(segment_ids_item, (0, max_len_dict['segment_ids']-segment_ids_item.shape[1]), mode='constant', value=0)]

        # if self.args.dep_embed == 'bert':
        # dep_tag_ids_item = pad_sequence(dep_tag_ids_item, batch_first=True, padding_value=0)
        # new_dep_tag_ids += [F.pad(dep_tag_ids_item, (0, 0, 0, max_len_dict['dep_tag_ids']-dep_tag_ids_item.shape[1]), mode='constant', value=0)]
        # elif self.args.dep_embed == 'combine':
        dep_tag_ids_item = [F.pad(ditem, (0, max_len_dict['dep_tag_ids']-ditem.shape[-1], 0, max_len_dict['dep_tag_ids_outer']-ditem.shape[-2]), mode='constant', value=0) for ditem in dep_tag_ids_item]
        new_dep_tag_ids += [pad_sequence(dep_tag_ids_item, batch_first=True, padding_value=0)]

        new_SPN += [F.pad(spn_item, (0, max_len_dict['input_cat_ids']-spn_item.shape[1]), mode='constant', value=0)]

    input_aspect_ids = pad_sequence(new_input_aspect_ids, batch_first=True, padding_value=0)
    input_cat_ids = pad_sequence(new_input_cat_ids, batch_first=True, padding_value=0)
    segment_ids = pad_sequence(new_segment_ids, batch_first=True, padding_value =0)
    aspect_indexer = pad_sequence(new_aspect_indexer, batch_first=True, padding_value=1)
    dep_tag_ids = pad_sequence(new_dep_tag_ids, batch_first=True, padding_value=0)
    SPN = pad_sequence(new_SPN, batch_first=True, padding_value=0)

    # Pad sequences.
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    # indexer are padded with 1, for ...
    word_indexer = pad_sequence(word_indexer, batch_first=True, padding_value=1)

    # aspect_positions = pad_sequence(
    #     aspect_positions, batch_first=True, padding_value=0)

    dep_dir_ids = pad_sequence(dep_dir_ids, batch_first=True, padding_value=0)
    pos_class = pad_sequence(pos_class, batch_first=True, padding_value=0)

    dep_rel_ids = pad_sequence(dep_rel_ids, batch_first=True, padding_value=0)
    # dep_heads = pad_sequence(dep_heads, batch_first=True, padding_value=0)
    aspect_dep_tag_ids = pad_sequence(aspect_dep_tag_ids, batch_first=True, padding_value=0)
    sentiment = pad_sequence(sentiment, batch_first=True, padding_value=10)
    aspect_relation = pad_sequence(aspect_relation, batch_first=True, padding_value=10)

    # Sort all tensors based on text len.
    _, sorted_idx = text_len.sort(descending=True)
    input_ids = input_ids[sorted_idx]
    input_aspect_ids = input_aspect_ids[sorted_idx]
    word_indexer = word_indexer[sorted_idx]
    aspect_indexer = aspect_indexer[sorted_idx]
    input_cat_ids = input_cat_ids[sorted_idx]
    segment_ids = segment_ids[sorted_idx]
    # aspect_positions = aspect_positions[sorted_idx]
    dep_tag_ids = dep_tag_ids[sorted_idx]
    dep_dir_ids = dep_dir_ids[sorted_idx]
    pos_class = pos_class[sorted_idx]
    text_len = text_len[sorted_idx]
    aspect_len = aspect_len[sorted_idx]
    sentiment = sentiment[sorted_idx]
    aspect_relation = aspect_relation[sorted_idx]
    dep_rel_ids = dep_rel_ids[sorted_idx]
    # dep_heads = dep_heads[sorted_idx]
    aspect_dep_tag_ids = aspect_dep_tag_ids[sorted_idx]
    SPN = SPN[sorted_idx]

    # return input_ids, word_indexer, input_aspect_ids, aspect_indexer,input_cat_ids,segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions, dep_dir_ids
    return input_ids, word_indexer, input_aspect_ids, aspect_indexer, input_cat_ids, segment_ids, aspect_relation, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_dir_ids, aspect_dep_tag_ids, SPN
