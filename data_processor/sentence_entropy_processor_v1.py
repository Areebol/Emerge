from .base_processor import BaseProcessor
from utils.meter import AverageMeter
from utils.process_data import *
import torch
import os
import csv
import logging

class v1SentenceEntropyProcessor(BaseProcessor):
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "SentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = True
        self.sentence_size = None
        self.is_column = False
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate,split_words=None):
        logging.info(f"{self.name} process data")
        res = model_generate['generate']
        input_ids = model_generate['input_ids']
        encoder = get_encoder_k(self.model,-1)

        # 按照句子切分
        if self.sentence_size == None:
            # 划分输入        
            if split_words:
                split_tokens = split_sentence(tokenizer=self.tokenizer,question=data,input_ids=input_ids,split_words=split_words)
            else:
                split_tokens = split_sentence(tokenizer=self.tokenizer,question=data,input_ids=input_ids)
        # 按照固定大小切分
        else:
            split_tokens = torch.tensor([i for i in range(0,input_ids.shape[1],self.sentence_size)])
            logging.info(f"split_tokens = {split_tokens}")
        
        # 根据句子切分attention矩阵 weight权重，token_ids 权重对应token下标
        # 固定不归一化
        weights,token_ids = split_attn_matrix(self.model,res,split_tokens,soft_max=False,is_column=self.is_column) # 控制是否按列求子矩阵
        
        # 加权计算embedding得到hidden_states
        hidden_states = weighted_hidden_states(weights,token_ids,res)        

        # 计算attention矩阵
        attn_matrix = get_attention_matrix(encoder,hidden_states,soft_max=self.soft_max).squeeze(0).to(torch.float32)
        
        # 计算entropy
        with torch.no_grad():
            sentence_entropy = get_attention_entropy(attn_matrix.cpu(),soft_max=self.soft_max,avg_head=self.avg_head)
            if self.avg_head == False:
                mean_sentence_entropy = torch.mean(sentence_entropy,dim=0).squeeze()
            else:
                mean_sentence_entropy = sentence_entropy.squeeze(0)
        return mean_sentence_entropy.tolist()

    def append_data_to_csv(self, data):
        with open(self.save_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in data:
                writer.writerow([key, value])

    def save_data(self, index, data):
        self.append_data_to_csv([(index,data)])
    
    def set(self, data_type):
        self.data_type = data_type
        self.save_dir = self.exp_dir + f"/{self.name}" + f"/{self.data_type}"
        BaseProcessor.set(self)
    
    def reset(self):
        self.total_entropy.reset()
 
 # V1 用方attn matrix计算权重       
class v1AvgHeadSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "v1AvgHeadSoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")
        
class v1SoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "v1SoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = False
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")

class v1UnSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "v1UnSoftMaxSentenceEntropyProcessor"
        self.soft_max = False
        self.avg_head = False
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")
        
# V1 用列attn matrix计算权重
class v1ColumnAvgHeadSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "v1ColumnAvgHeadSoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = True
        self.is_column = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")
        
class v1ColumnSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "v1ColumnSoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = False
        self.is_column = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")

class v1ColumnUnSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.name = "v1ColumnUnSoftMaxSentenceEntropyProcessor"
        self.soft_max = False
        self.avg_head = False
        self.is_column = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")

# V1 用方attn matrix计算权重 固定分句大小为 sentence_size
class v1FixedAvgHeadSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.sentence_size = 8
        self.name = f"v1Size{self.sentence_size}AvgHeadSoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")

class v1FixedSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.sentence_size = 8
        self.name = f"v1Size{self.sentence_size}SoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = False
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")

class v1FixedUnSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.sentence_size = 8
        self.name = f"v1Size{self.sentence_size}UnSoftMaxSentenceEntropyProcessor"
        self.soft_max = False
        self.avg_head = False
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")
        
# V1 用列attn matrix计算权重 固定分句大小为 sentence_size
class v1ColumnFixedAvgHeadSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.sentence_size = 8
        self.name = f"v1Size{self.sentence_size}ColumnAvgHeadSoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = True
        self.is_column = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")

class v1ColumnFixedSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.sentence_size = 8
        self.name = f"v1Size{self.sentence_size}ColumnSoftMaxSentenceEntropyProcessor"
        self.soft_max = True
        self.avg_head = False
        self.is_column = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")

class v1ColumnFixedUnSoftMaxSentenceEntropyProcessor(v1SentenceEntropyProcessor):
    def __init__(self, model, tokenizer, model_config):
        v1SentenceEntropyProcessor.__init__(self, model, tokenizer, model_config)
        self.sentence_size = 8
        self.name = f"v1Size{self.sentence_size}ColumnUnSoftMaxSentenceEntropyProcessor"
        self.soft_max = False
        self.avg_head = False
        self.is_column = True
        logging.info(f"Init {self.name} : [soft_max : {self.soft_max} avg_head : {self.avg_head}]")