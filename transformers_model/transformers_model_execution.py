##############################
## BERT - Model Execution
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


def classification_iterate(train_dataloader, validation_dataloader, optimizer,
                           scheduler, model, device, training_stats, epochs):
    start_time = time.time()
    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_train_loss = 0
        model.train()

        optimizer, scheduler, total_train_loss, total_train_accuracy = _train_epoch(
                                                             train_dataloader,
                                                             optimizer,
                                                             scheduler,
                                                             model,
                                                             total_train_loss,
                                                             device
                                                             )
        _print_gpu_space()
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)

        print(" Average training loss: {0:.2f}".format(avg_train_loss))
        print(" Average training accuracy: {0:.2f}".format(avg_train_accuracy))
        print("")
        print("Running Validation...")
        model.eval()

        total_eval_accuracy, total_eval_loss = _validate_epoch(
                                          validation_dataloader, model, device)
        _print_gpu_space()
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)

        print(" Accuracy: {0:.2f}".format(avg_val_accuracy))
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        print(" Validation Loss: {0:.2f}".format(avg_val_loss))

        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'training_loss': avg_train_loss,
                'training_accuracy': avg_train_accuracy,
                'validation_loss': avg_val_loss,
                'validation_accuracy': avg_val_accuracy
            }
        )
    end_time = time.time()
    training_stats.append({'training_time': start_time - end_time})
    print("Training complete!")

    return (training_stats, model)


def _train_epoch(train_dataloader, optimizer, scheduler, model, total_train_loss, device):
    total_train_accuracy = 0
    for step, batch in enumerate(train_dataloader):
        b_tokenized_sentences = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].type(torch.LongTensor).to(device)
        model.zero_grad()
        loss, logits = model(b_tokenized_sentences,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
        total_train_loss += loss.item()
        loss.backward()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_train_accuracy += _flat_accuracy(logits, label_ids)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        del b_tokenized_sentences
        del b_input_mask
        del b_labels
    return (optimizer, scheduler, total_train_loss, total_train_accuracy)


def _validate_epoch(validation_dataloader, model, device):
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in validation_dataloader:
        b_tokenized_sentences = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].type(torch.LongTensor).to(device)
        with torch.no_grad():        
            (loss, logits) = model(b_tokenized_sentences,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += _flat_accuracy(logits, label_ids)
        del b_tokenized_sentences
        del b_input_mask
        del b_labels
    return total_eval_accuracy, total_eval_loss


def _flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def _print_gpu_space():
    GPUs = GPU.getGPUs()
    # XXX: only one GPU on Colab and isn’t guaranteed
    gpu = GPUs[0]
    def printm():
        process = psutil.Process(os.getpid())
        print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available), " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
        print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    printm()


def run_model(model, prediction_dataloader, device):
    model.eval()
    predictions , true_labels = [], []

    for batch in prediction_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_tokenized_sentences, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_tokenized_sentences, token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)
    return flat_predictions, flat_true_labels

