from .base_processor import BaseProcessor
from utils.meter import AverageMeter
import os
import csv
import logging
import torch
from utils.process_data import get_attention_entropy
from utils.process_data import *

class AvgHeadSoftMaxMeanEntropyProcessor(BaseProcessor):
    """对多头的attention matrix求平均再计算熵值"""
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "AvgHeadSoftMaxMeanEntropyProcessor"
        logging.info(f"Init {self.name}")
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate,split_words=None):
        logging.info(f"{self.name} process data")
        res = model_generate['generate']
        encoder = get_encoder_k(self.model,-1)
        hidden_state = get_hidden_state_k(res,-2)
        # 手动计算最后一层的attention weight
        attentions = get_attention_matrix(encoder,hidden_state,soft_max=True).to(torch.float32).cpu() # shape = (bs_size,#heads,len,len)
        avg_attention= torch.mean(attentions,dim=1) # shape = (bs_size,len,len)
        res_entropy = get_attention_entropy(avg_attention) # shape = (bs_size,len)
        mean_entropy = res_entropy[:,1:].mean()
        self.total_entropy.update(mean_entropy)
        logging.info(mean_entropy.item())
        return mean_entropy.item()
    
    def append_data_to_csv(self, data):
        with open(self.save_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in data:
                writer.writerow([key, value])

    def save_data(self, index, data):
        self.append_data_to_csv([(index,data)])
    
    def set(self, data_type):
        self.data_type = data_type
        self.save_dir = self.exp_dir + "/AvgHeadSoftMaxMeanEntropy" + f"/{self.data_type}"
        BaseProcessor.set(self)
    
    def reset(self):
        self.total_entropy.reset()
        
class AvgHeadUnSoftMaxMeanEntropyProcessor(BaseProcessor):
    """对多头的attention matrix求平均再计算熵值"""
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "AvgHeadUnSoftMaxMeanEntropyProcessor"
        logging.info(f"Init f{self.name}")
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate,split_words=None):
        logging.info(f"{self.name} process data")
        res = model_generate['generate']
        encoder = get_encoder_k(self.model,-1)
        hidden_state = get_hidden_state_k(res,-2)
        # 手动计算最后一层的attention weight
        attentions = get_attention_matrix(encoder,hidden_state,soft_max=False).to(torch.float32).cpu() # shape = (bs_size,#heads,len,len)
        avg_attention= torch.mean(attentions,dim=1) # shape = (bs_size,len,len)
        res_entropy = get_attention_entropy(avg_attention) # shape = (bs_size,len)
        mean_entropy = res_entropy[:,1:].mean()
        self.total_entropy.update(mean_entropy)
        logging.info(mean_entropy.item())
        return mean_entropy.item()

    def append_data_to_csv(self, data):
        with open(self.save_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in data:
                writer.writerow([key, value])

    def save_data(self, index, data):
        self.append_data_to_csv([(index,data)])
    
    def set(self, data_type):
        self.data_type = data_type
        self.save_dir = self.exp_dir + "/AvgHeadUnSoftMaxMeanEntropy" + f"/{self.data_type}"
        BaseProcessor.set(self)
    
    def reset(self):
        self.total_entropy.reset()