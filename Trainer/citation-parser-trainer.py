'''
    This is used for training deep learning model
'''

#!/usr/bin/env python
# coding: utf-8


import os
import time
import gensim
from collections import Counter
import torch
from torch import nn
from torch.optim import Adam
from torch.optim import AdamW
from spacy.lang.id import Indonesian
import gensim.models.keyedvectors as word2vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

import math
import time
import gensim
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam

from torchtext.legacy.data import Field, NestedField, BucketIterator
from torchtext.legacy.datasets import SequenceTaggingDataset
from torchtext.legacy.vocab import Vocab
from torchcrf import CRF
from collections import Counter
from spacy.lang.id import Indonesian
import logging
import gc
import torch.utils
import torch.utils.checkpoint

import torch.nn.utils.prune as prune
import torch.nn.functional as F

# Set seed for reproducibility
torch.manual_seed(123)

DRIVE_ROOT = "/home/muddi004/muddi/citationParser/data"

timestr = time.strftime("%Y%m%d-%H%M%S")
log_directory = "/home/muddi004/muddi/citationParser/log/"
log_file = log_directory + 'multigpu-train' +'.log'
logging.basicConfig(filename=log_file,level=logging.DEBUG)

available_gpu = torch.cuda.is_available()
if available_gpu:
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU is available: {torch.cuda.get_device_name(0)}")
    use_device = torch.device("cuda")
else:
    use_device = torch.device("cpu")

from corpus import Corpus

start = time.time()
print('Start Reading the Datasets')
logging.info('Start Reading the Datasets')
corpus = Corpus(
    input_folder=f"{DRIVE_ROOT}",
    min_word_freq=3,
    batch_size=64,
    wv_file = '/home/muddi004/muddi/citationParser/data/GoogleNews-vectors-negative300.bin'
)
print(f"Train set: {len(corpus.train_dataset)} sentences")
print(f"Val set: {len(corpus.val_dataset)} sentences")
print(f"Test set: {len(corpus.test_dataset)} sentences")
logging.info(f"Train set: {len(corpus.train_dataset)} sentences")
logging.info(f"Val set: {len(corpus.val_dataset)} sentences")
logging.info(f"Test set: {len(corpus.test_dataset)} sentences")
end = time.time()
print("Total time required: {}".format(end - start))
logging.info("Total time required: {}".format(end - start))

from transformerModel import Transformer, PositionalEncoding

transformer = Transformer(
    input_dim=len(corpus.word_field.vocab),
    embedding_dim=300,
    char_emb_dim=25, #37,  # NEWLY MODIFIED: TRANSFORMER
    char_input_dim=len(corpus.char_field.vocab),
    char_cnn_filter_num=4,  # NEWLY MODIFIED: TRANSFORMER
    char_cnn_kernel_size=3,
    char_lstm_hidden=50,
    attn_heads=16,  # NEWLY MODIFIED: TRANSFORMER
    fc_hidden=200,  # NEWLY MODIFIED: TRANSFORMER // 256
    trf_layers=1,
    output_dim=len(corpus.tag_field.vocab),
    emb_dropout=0.5,
    cnn_dropout=0.25,
    lstm_dropout = 0.2,
    trf_dropout=0.1,  # NEWLY MODIFIED: TRANSFORMER
    fc_dropout=0.25,
    word_pad_idx=corpus.word_pad_idx,
    char_pad_idx=corpus.char_pad_idx,
    tag_pad_idx=corpus.tag_pad_idx,
    device=use_device
)
transformer.init_embeddings(
    pretrained=corpus.word_field.vocab.vectors if corpus.wv_model else None,
    freeze=True
)

print(f"The model has {transformer.count_parameters():,} trainable parameters.")
logging.info(f"The model has {transformer.count_parameters():,} trainable parameters.")
print(transformer)

if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    transformer = nn.DataParallel(transformer)

transformer.to(use_device)

###############################################################################
# Training code
###############################################################################

class Trainer(object):

    def __init__(self, model, data, optimizer_cls, device):  # NEWLY MODIFIED: GPU
        self.device = device  # NEWLY ADDED: GPU
        self.model = model.to(self.device)  # NEWLY MODIFIED: GPU
        self.data = data
        self.optimizer = optimizer_cls(model.parameters(), lr=0.001)

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def accuracy(self, preds, y):
        flatten_preds = [pred for sent_pred in preds for pred in sent_pred]
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        correct = [pred == tag for pred, tag in zip(flatten_preds, flatten_y)]
        
        del flatten_preds
        del flatten_y
        
        return sum(correct) / len(correct) if len(correct) > 0 else 0

    def epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()        
       
        # batch accumulation parameter
        '''
        accum_iter = 16  
        '''
        
        #self.data.train_iter.create_batches()
        #for batch_idx, batch in enumerate(self.data.train_iter):
        for batch in self.data.train_iter:            
            self.model.zero_grad(set_to_none=True)
            
            # words = [sent len, batch size]
            words = batch.word.to(self.device)  # NEWLY MODIFIED: GPU            
            print("Sent Len from words: {}".format(words.size(dim=0)))
            if words.size(dim=0) > 200: # FIX For: Extra lengthy sentence - Skip that from training!
                pass
            else:
                # chars = [batch size, sent len, char len]
                chars = batch.char.to(self.device)  # NEWLY MODIFIED: GPU
                # tags = [sent len, batch size]
                true_tags = batch.tag.to(self.device)  # NEWLY MODIFIED: GPU
                #print("true_tags:\n {}".format(true_tags))

                pred_tags_list, batch_loss = self.model(words, chars, true_tags)
                # to calculate the loss and accuracy, we flatten true tags
                true_tags_list = [
                    [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                    for sent_tag in true_tags.permute(1, 0).tolist()
                ]
                batch_acc = self.accuracy(pred_tags_list, true_tags_list)

                # normalize loss to account for batch accumulation
                '''
                batch_loss = batch_loss / accum_iter 
                '''

                batch_loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) # clipping_value = 1

                '''
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(self.data.train_iter)):
                    self.optimizer.step()

                    epoch_loss += float(batch_loss.detach())
                    epoch_acc += float(batch_acc)                       

                    del batch_acc
                    del batch_loss
                    del words
                    del chars
                    del true_tags
                    del pred_tags_list
                '''               
                self.optimizer.step()
                epoch_loss += float(batch_loss.detach())
                epoch_acc += float(batch_acc)                       

                del batch_acc
                del batch_loss
                del words
                del chars
                del true_tags
                del pred_tags_list                
            
        #  clearing the occupied cuda memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                words = batch.word.to(self.device)  # NEWLY MODIFIED: GPU
                chars = batch.char.to(self.device)  # NEWLY MODIFIED: GPU
                true_tags = batch.tag.to(self.device)  # NEWLY MODIFIED: GPU
                pred_tags, batch_loss = self.model(words, chars, true_tags)
                true_tags_list = [
                    [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                    for sent_tag in true_tags.permute(1, 0).tolist()
                ]
                batch_acc = self.accuracy(pred_tags, true_tags_list)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self, n_epochs):
        prev_train_loss = float('inf') 
        for epoch in range(n_epochs):
            start_time = time.time()
            
            # Creates once at the beginning of training
            #scaler = torch.cuda.amp.GradScaler()
            train_loss, train_acc = self.epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = Trainer.epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")

            val_loss, val_acc = self.evaluate(self.data.val_iter)
            print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")
            
            # Save the best model
            if train_loss is not None and prev_train_loss > train_loss:
                prev_train_loss = train_loss
                self.save()
            
            del train_loss
            del train_acc
            del val_loss
            del val_acc

        '''    
        test_loss, test_acc = self.evaluate(self.data.test_iter)
        print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")
        logging.info(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")
        '''
    
    def save(self):
        saving_directory = r"/home/muddi004/muddi/citationParser/savedmodel/342K-v2/"
        
        path_model = os.path.join(saving_directory, "transformermodel.pth")
        torch.save(self.model.state_dict(), path_model)
        
        '''
            Save vocabulary
        '''
        path_word_vocab = os.path.join(saving_directory, "word_field_obj.pth")
        torch.save(corpus.word_field, path_word_vocab)
        
        path_tag_vocab = os.path.join(saving_directory, "tag_field_obj.pth")
        torch.save(corpus.tag_field, path_tag_vocab)
        
        path_char_vocab = os.path.join(saving_directory, "char_field_obj.pth")
        torch.save(corpus.char_field, path_char_vocab)

        
    def infer(self, sentence, tokens, true_tags=None):
        self.model.eval()
        # tokenize sentence
        nlp = Indonesian()
        #tokens = [token.text for token in nlp(sentence)]
        tokens = tokens
        max_word_len = max([len(token) for token in tokens])
        # transform to indices based on corpus vocab
        numericalized_tokens = [self.data.word_field.vocab.stoi[token.lower()] for token in tokens]
        numericalized_chars = []
        char_pad_id = self.data.char_pad_idx
        for token in tokens:
            numericalized_chars.append(
                [self.data.char_field.vocab.stoi[char] for char in token]
                + [char_pad_id for _ in range(max_word_len - len(token))]
            )
        # find unknown words
        unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        # begin prediction
        token_tensor = torch.as_tensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1).to(self.device)  # NEWLY MODIFIED: GPU
        char_tensor = torch.as_tensor(numericalized_chars)
        char_tensor = char_tensor.unsqueeze(0).to(self.device)  # NEWLY MODIFIED: GPU
        predictions, _ = self.model(token_tensor, char_tensor)
        # convert results to tags
        predicted_tags = [self.data.tag_field.vocab.itos[t] for t in predictions[0]]
        # print inferred tags
        max_len_token = max([len(token) for token in tokens] + [len('word')])
        max_len_tag = max([len(tag) for tag in predicted_tags] + [len('pred')])

        print(
            f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
            + ("\ttrue tag" if true_tags else "")
        )
        logging.info(
            f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
            + ("\ttrue tag" if true_tags else "")
        )
        for i, token in enumerate(tokens):
            is_unk = "✓" if token in unks else ""
            print(
                f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
                + (f"\t{true_tags[i]}" if true_tags else "")
            )
            logging.info(
                f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
                + (f"\t{true_tags[i]}" if true_tags else "")
            )

        return predicted_tags

#from trainer import Trainer

print('Start Training')
logging.info('Start Training')
trainer = Trainer(
    model=transformer,
    data=corpus,
    optimizer_cls=Adam,
    device=use_device,
)
trainer.train(25) 

# Save the model
#modelStorage = r"/home/muddi004/muddi/citationParser/savedmodel/100K/"
#trainer.save(modelStorage)

''' 
Changable lines:
- trainer.train(5) 
- modelStorage location
- on corpus file : training/testing dataset location
'''