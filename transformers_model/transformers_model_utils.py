##############################
## BERT - Model
##############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import time
import datetime
import io
import psutil
import humanize
import os
import GPUtil as GPU
import gc


from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_model(bert_model='bert-base-uncased', num_labels=2):
    model = BertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=bert_model,
        num_labels=num_labels,
        output_attentions = False,
        output_hidden_states = False
    )
    model.cuda()
    return model


def get_data_loader(batch_size, train_dataset, val_dataset):
    train_dataloader = DataLoader(
              train_dataset,
              sampler = RandomSampler(train_dataset),
              batch_size = batch_size
          )

    validation_dataloader = DataLoader(
              val_dataset,
              sampler = SequentialSampler(val_dataset),
              batch_size = batch_size
          )
    return train_dataloader, validation_dataloader


def get_adam_optimizer(model, learning_rate=2e-5, epsilon=1e-8, weight_decay=0):
  optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon,
                  weight_decay=weight_decay
                )
  return optimizer


def load_test_set(df):
    sentences = df['comment'].to_list()
    labels = df['label'].to_list()

    tokenized_sentences = []
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,
                            add_special_tokens = True,
                            max_length = 250,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            truncation=True
                    )
        tokenized_sentences.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    tokenized_sentences = torch.cat(tokenized_sentences, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    batch_size = 32  

    prediction_data = TensorDataset(tokenized_sentences, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler,
                                       batch_size=batch_size)
    return prediction_dataloader

