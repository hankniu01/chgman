import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, multilabel_confusion_matrix, recall_score, precision_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from datasets import my_collate_bert_for_overall
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer

import time
from time import strftime, localtime
import pickle

# args.logger = logging.getargs.logger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove' or embedding_type == 'elmo':
        # sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {  'sentence': batch[0],
                    'aspect': batch[1], # aspect token
                    'dep_tags': batch[2], # reshaped
                    'pos_class': batch[3],
                    'text_len': batch[4],
                    'aspect_len': batch[5],
                    'dep_rels': batch[7], # adj no-reshape
                    'dep_heads': batch[8],
                    'aspect_position': batch[9],
                    'dep_dirs': batch[10]
                    }
        labels = batch[6]
    else: # bert
        # input_ids, word_indexer, input_aspect_ids, aspect_indexer, input_cat_ids, segment_ids, 
        # aspect_relation, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_dir_ids, aspect_dep_tag_ids
        inputs = {  'input_ids': batch[0],
                    'input_aspect_ids': batch[2],
                    'word_indexer': batch[1],
                    'aspect_indexer': batch[3],
                    'input_cat_ids': batch[4],
                    'segment_ids': batch[5],
                    # 'aspect_relation': batch[6]
                    'dep_tags': batch[7],
                    'pos_class': batch[8],
                    'text_len': batch[9],
                    'aspect_len': batch[10],
                    'dep_rels': batch[12],
                    # 'dep_heads': batch[12],
                    # 'aspect_position': batch[13],
                    'dep_dirs': batch[13],
                    'aspect_dep_tag_ids': batch[14],
                    'SPN': batch[15]}
        labels = {}
        labels['labels'] = batch[11]
        labels['relation_labels'] = batch[6]
    return inputs, labels


def get_collate_fn(args):
    return my_collate_bert_for_overall

def adjust_lr(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=3, num_training_steps=20)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return optimizer, scheduler


def train_overall(args, train_dataset, test_dataset, model):
    '''Train the model and to get gaussian case'''
    launchTimestamp = time.asctime(time.localtime(time.time()))

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn, drop_last=False)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    
    if args.embedding_type == 'bert':
        optimizer, scheduler = get_bert_optimizer(args, model)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # Train
    args.logger.info("***** Running training *****")
    args.logger.info("  Num examples = %d", len(train_dataset))
    args.logger.info("  Num Epochs = %d", args.num_train_epochs)
    args.logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    args.logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    args.logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    tr_loss, tr_sentiment_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    preds = None
    out_label_ids = None
    train_results = {}
    Cases = {}
    for epochID in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            inputs, Labels = get_input_from_batch(args, batch)
            labels = Labels['labels']
            # relation_labels = Labels['relation_labels']
            logit = model(**inputs)
            logit_ = logit.view(-1, logit.shape[-1])
            labels_ = labels.view(-1, 1).squeeze(1)
            sentiment_loss = F.cross_entropy(logit_, labels_, ignore_index=10)
            loss = sentiment_loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                sentiment_loss = sentiment_loss / args.gradient_accumulation_steps
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            # optimizer.step()

            tr_loss += loss.item()
            tr_sentiment_loss += sentiment_loss

            mask = labels.lt(10)
            num_class = logit.shape[-1]
            mask_3d = torch.repeat_interleave(mask.unsqueeze(-1), num_class, dim=-1)
            labels = torch.masked_select(labels, mask)
            logit = torch.masked_select(logit, mask_3d).view(-1, num_class)
            if preds is None:
                preds = logit.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logit.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0)

                
            # if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

                # Log metrics
        args.logger.info('total train loss {}'.format(tr_loss / global_step))
        args.logger.info('train sentiment loss {}'.format(tr_sentiment_loss / global_step))
        preds_ = np.argmax(preds, axis=1)
        train_result = compute_metrics(preds_, out_label_ids, epochID)
        train_results.update(train_result)
        for key in sorted(train_result.keys()):
            args.logger.info("Epoch: %s, train_acc_and_f1:  %s = %s", str(epochID), key, str(train_result[key]))

        ##########Eval################
        results = evaluate_for_overall(args, test_dataset, model, epochID)
        if all_eval_results != []:
            if results['acc'] > all_eval_results[-1]['acc'] or results['acc'] > 0.85:
            # if results['acc'] > 0.86:
                torch.save({'epoch': epochID, 'state_dict': model.state_dict(), 'best_loss': loss,
                    'optimizer': optimizer.state_dict(), 'acc': results['acc'], 'f1': results['f1']}, 
                    args.base_dir + '/' + args.checkpoint_path + args.multi_sub_dir + '/m-' + str(launchTimestamp) + '-' + str("%.4f" % loss) + '_acc_' + str("%.4f" % results['acc']) + '_f1_' +str("%.4f" % results['f1']) + '.pth.tar')


        all_eval_results.append(results)

        if args.is_lr_decay:
            adjust_lr(optimizer, args.learning_rate / (1 + epochID * args.lr_decay))
        if args.max_steps > 0 and global_step > args.max_steps:
            # epoch_iterator.close()
            break

    return global_step, tr_loss/global_step, all_eval_results



#######################evaluate##########################
def evaluate_for_overall(args, eval_dataset, model, epochID):
    results = {}
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn, drop_last=True)

    # Eval
    args.logger.info("***** Running evaluation *****")
    args.logger.info("  Num examples = %d", len(eval_dataset))
    args.logger.info("  Batch size = %d", args.eval_batch_size)
    Tmp_eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch if not isinstance(t, tuple))
        # batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            # inputs, labels = get_input_from_batch(args, batch)
            inputs, Labels = get_input_from_batch(args, batch)
            labels = Labels['labels']
            # relation_labels = Labels['relation_labels']
            logit = model(**inputs)
            logit_ = logit.view(-1, logit.shape[-1])
            labels_ = labels.view(-1, 1).squeeze(1)
            tmp_eval_loss = F.cross_entropy(logit_, labels_, ignore_index=10)

            Tmp_eval_loss += tmp_eval_loss.mean().item()

            mask = labels.lt(10)
            num_class = logit.shape[-1]
            mask_3d = torch.repeat_interleave(mask.unsqueeze(-1), num_class, dim=-1)
            labels = torch.masked_select(labels, mask)
            logit = torch.masked_select(logit, mask_3d).view(-1, num_class)

        nb_eval_steps += 1
        if preds is None:
            preds = logit.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logit.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

    Tmp_eval_loss = Tmp_eval_loss / nb_eval_steps

    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids, epochID)
    results.update(result)

    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(output_eval_file, 'a+') as writer:
        args.logger.info('***** Eval results *****')
        args.logger.info("  sentiment eval loss: %s", str(Tmp_eval_loss))
        for key in sorted(result.keys()):
            args.logger.info("  %s = %s", key, str(result[key]))
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    return results



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1_for_confusion_matirix(preds, labels, epochID):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    confusion = multilabel_confusion_matrix(labels, preds)
    TN, FN, TP, FP = confusion[:, 0, 0], confusion[:, 1, 0], confusion[:, 1, 1], confusion[:, 0, 1]
    return {
        "epochID": epochID,
        "acc": acc,
        "f1": f1, 
        "recall": recall, 
        "precision": precision,
        "confusion": confusion,
        "TN": TN, 
        "FN": FN, 
        "TP": TP,
        "FP": FP
    }

def acc_and_f1(preds, labels, epochID):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "epochID": epochID,
        "acc": acc,
        "f1": f1
    }

def compute_metrics(preds, labels, epochID):
    return acc_and_f1(preds, labels, epochID)
