'''
    This is used for training deep learning model
'''
import torch
import torch.nn as nn
import gc
import torch.utils
import torch.utils.checkpoint

import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torchcrf import CRF
from collections import Counter

from torch.optim import Adam
from torch.optim import AdamW
import math
import re

#Reproducing same results
SEED = 2019

#Torch
torch.manual_seed(SEED)

'''
    Version from pytorch official
'''
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Transformer(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 char_emb_dim,
                 char_input_dim,
                 char_cnn_filter_num,
                 char_cnn_kernel_size,
                 char_lstm_hidden,
                 attn_heads,
                 fc_hidden,
                 trf_layers,
                 output_dim,
                 emb_dropout,
                 cnn_dropout,
                 lstm_dropout,
                 trf_dropout,
                 fc_dropout,
                 word_pad_idx,
                 char_pad_idx,
                 tag_pad_idx,
                 device): # NEWLY ADDED: GPU  
        super().__init__()
        self.char_pad_idx = char_pad_idx
        self.word_pad_idx = word_pad_idx
        self.tag_pad_idx = tag_pad_idx
        self.device = device  # NEWLY ADDED: GPU
        # LAYER 1A: Word Embedding
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        # LAYER 1B: Char Embedding-CNN
        self.char_emb_dim = char_emb_dim
        self.char_emb = nn.Embedding(
            num_embeddings=char_input_dim,
            embedding_dim=char_emb_dim,
            padding_idx=char_pad_idx
        )
        
        self.char_lstm_hidden = char_lstm_hidden
        self.char_lstm = nn.LSTM(char_emb_dim, # 25
                                 hidden_size=char_lstm_hidden, # 50 
                                 bidirectional=True,
                                 batch_first=True)
        '''
        self.char_cnn = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=char_emb_dim * char_cnn_filter_num, # 25*4
            kernel_size=char_cnn_kernel_size,
            groups=char_emb_dim  # different 1d conv for each embedding dim
        )
        
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        '''
        self.lstm_dropout = nn.Dropout(lstm_dropout)
        ### BEGIN MODIFIED SECTION: TRANSFORMER ###
        # LAYER 2: Transformer
        #all_emb_size = embedding_dim + (char_emb_dim * char_cnn_filter_num)
        all_emb_size = embedding_dim + (char_lstm_hidden * 2)
        self.position_encoder = PositionalEncoding(
            d_model=all_emb_size
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=all_emb_size,
            nhead=attn_heads,
            activation="relu",
            dropout=trf_dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=trf_layers
        )
        # LAYER 3: 2-layers fully-connected with GELU activation in-between
        '''     
        self.fc1 = nn.Linear(
            in_features=all_emb_size,
            out_features=fc_hidden
        )
        self.fc1_gelu = nn.GELU()
        
        '''
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=all_emb_size,out_features=fc_hidden),
            nn.GELU()
        )
   
        self.fc1_norm = nn.LayerNorm(fc_hidden)
        
        self.fc2_dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(
            in_features=fc_hidden,
            out_features=output_dim
        )
        ### END MODIFIED SECTION ###
        # LAYER 4: CRF
        self.crf = CRF(num_tags=output_dim)
        # init weights from normal distribution
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)
            
    def custom_encoder_forward(self,module):
        def custom_forward(*inputs):
            output = module(inputs[0],src_key_padding_mask=inputs[1])
            return output
        return custom_forward
    
    def custom_linear_forward(self, module):
        def custom_forward(*inputs):
            output = module(inputs[0])
            return output
        return custom_forward
    
    def custom_charlstm_forward(self,module):
        def custom_forward(*inputs):
            output = module(inputs[0])
            return output
        return custom_forward
    
    def custom_crf_forward(self, module):
        def custom_forward(*inputs):
            output = module(inputs[0],tags=inputs[1], mask=inputs[2])
            return output
        return custom_forward

    def forward(self, words, chars, tags=None):
        # words = [sentence length, batch size]
        # chars = [batch size, sentence length, word length)
        # tags = [sentence length, batch size]
        # embedding_out = [sentence length, batch size, embedding dim]
        #print('Words : {}'.format(words.shape))
        embedding_out = self.emb_dropout(self.embedding(words))
        # character cnn layer forward
        # reference: https://github.com/achernodub/targer/blob/master/src/layers/layer_char_cnn.py
        # char_emb_out = [batch size, sentence length, word length, char emb dim]
        char_emb_out = self.emb_dropout(self.char_emb(chars))
        batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
        #print("batch_size: {}, sent_len: {}, word_len: {}, char_emb_dim: {}".format(batch_size, sent_len, word_len, char_emb_dim))

        #if sent_len < 300:
        char_vec = torch.zeros(char_emb_out.shape[0], char_emb_out.shape[1], self.char_lstm_hidden * 2, device=self.device)
        for idx, ch in enumerate(char_emb_out):
            '''            
            s_ch_rep, _ = self.char_lstm(ch)
            '''
            s_ch_rep, _ = torch.utils.checkpoint.checkpoint(self.custom_charlstm_forward(self.char_lstm), ch)

            s_ch_rep_f = s_ch_rep[:, -1, 0: self.char_lstm_hidden]
            s_ch_rep_b = s_ch_rep[:, 0, self.char_lstm_hidden:]
            s_ch_rep = torch.cat((s_ch_rep_f, s_ch_rep_b), dim=1)
            char_vec[idx] = s_ch_rep
        
        char_lstm = self.lstm_dropout(char_vec)  
        #print("char_lstm {}".format(char_lstm.shape))
        # concat word and char embedding
        # char_lstm_p = [sentence length, batch size, char hidden dim * 2]
        char_lstm_p = char_lstm.permute(1, 0, 2)
        
        #print('Word embedding shape: {} | char_lstm_p shape: {}'.format(embedding_out.shape,char_lstm_p.shape))
        word_features = torch.cat((embedding_out, char_lstm_p), dim=2)

        ### BEGIN MODIFIED SECTION: TRANSFORMER ###
        # Transformer
        key_padding_mask = torch.as_tensor(words == self.word_pad_idx).permute(1, 0)
        # pos_out = [sentence length, batch size, embedding dim + char emb dim * num filter]
        #print('Word feature shape for position encoder input: {}'.format(word_features.shape))
        pos_out = self.position_encoder(word_features)
        # enc_out = [sentence length, batch size, embedding dim + char emb dim]
        '''        
        enc_out = self.encoder(pos_out, src_key_padding_mask=key_padding_mask)
        '''
        enc_out = torch.utils.checkpoint.checkpoint(self.custom_encoder_forward(self.encoder),pos_out,key_padding_mask) # Checkpointing

        #print("enc out {}".format(enc_out.shape))
        # Fully-connected
        # fc1_out = [sentence length, batch size, fc hidden]
        '''
        fc1_out = self.fc1_norm(self.fc1_gelu(self.fc1(enc_out)))
        '''
        fc1_out = torch.utils.checkpoint.checkpoint(self.fc1,enc_out)

        fc1_out = self.fc1_norm(fc1_out)
        #print("fc1_out {}".format(fc1_out.shape))
        # fc2_out = [sentence length, batch size, output dim]
        '''      
        fc2_out = self.fc2(self.fc2_dropout(fc1_out))
        '''
        fc2_out = torch.utils.checkpoint.checkpoint(self.custom_linear_forward(self.fc2),self.fc2_dropout(fc1_out))
  
        ### END MODIFIED SECTION ###
        # CRF
        crf_mask = words != self.word_pad_idx
        crf_out = self.crf.decode(fc2_out, mask=crf_mask)
        '''
        crf_loss = -self.crf(fc2_out, tags=tags, mask=crf_mask) if tags is not None else None
        '''
        crf_loss = -torch.utils.checkpoint.checkpoint(self.custom_crf_forward(self.crf),fc2_out,tags,crf_mask) if tags is not None else None

        return crf_out, crf_loss

    def init_embeddings(self, pretrained=None, freeze=True):
        # initialize embedding for padding as zero
        self.embedding.weight.data[self.word_pad_idx] = torch.zeros(self.embedding_dim)
        self.char_emb.weight.data[self.char_pad_idx] = torch.zeros(self.char_emb_dim)
        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(pretrained),
                padding_idx=self.word_pad_idx,
                freeze=freeze
            )
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)