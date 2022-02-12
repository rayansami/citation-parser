'''
    This class loads the training dataset and creates embeddings
'''

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

class Corpus(object):

    def __init__(self, input_folder, min_word_freq, batch_size, wv_file=None):
        # list all the fields
        self.word_field = Field(lower=True)  # [sent len, batch_size]
        self.tag_field = Field(unk_token=None, is_target=True)  # [sent len, batch_size]
        # Character-level input
        self.char_nesting_field = Field(tokenize=list)
        self.char_field = NestedField(self.char_nesting_field)  # [batch_size, sent len, max len char]
        # create dataset using built-in parser from torchtext
        self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
            path=input_folder,
            train="GIANT-1564-v2-train.txt",
            validation="data-test.txt",
            test="data-test.txt",
            fields=(
                (("word", "char"), (self.word_field, self.char_field)),
                ("tag", self.tag_field)
            )
        )
       
        # convert fields to vocabulary list
        if wv_file:
            #self.wv_model = gensim.models.word2vec.Word2Vec.load(wv_file)
            self.wv_model = word2vec.KeyedVectors.load_word2vec_format(wv_file, binary=True)
            self.embedding_dim = self.wv_model.vector_size
            word_freq = {word: self.wv_model.wv.vocab[word].count for word in self.wv_model.wv.vocab}
            word_counter = Counter(word_freq)
            self.word_field.vocab = Vocab(word_counter, min_freq=min_word_freq)
            vectors = []
            
            for word, idx in self.word_field.vocab.stoi.items():
                if word in self.wv_model.wv.vocab.keys():
                    vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
                else:
                    vectors.append(torch.zeros(self.embedding_dim))
            self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                vectors=vectors,
                dim=self.embedding_dim
            )
        else:
            self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
        # build vocab for tag and characters
        self.char_field.build_vocab(self.train_dataset.char)
        self.tag_field.build_vocab(self.train_dataset.tag)
        # create iterator for batch input
        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
            batch_size=batch_size
        )
        
        # prepare padding index to be ignored during model training/evaluation
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]