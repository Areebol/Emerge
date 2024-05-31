from .base_processor import BaseProcessor
from utils.meter import AverageMeter
import os
import csv
import logging

class MeanEntropyProcessor(BaseProcessor):
    def __init__(self, model, tokenizer, model_config):
        BaseProcessor.__init__(self, model, tokenizer, model_config)
        logging.info("Init MeanEntropyProcessor")
        self.total_entropy = AverageMeter()

    def process_data(self, index, data, model_generate):
        res_entropy = model_generate['entropy']
        num_input_tokens = res_entropy[0].__len__()
        num_heads = res_entropy.shape[0]
        mean_entropy = res_entropy[:,:].mean()
        self.total_entropy.update(mean_entropy)
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
        self.save_dir = self.exp_dir + "/MeanEntropy" + f"/{self.data_type}"
        BaseProcessor.set(self)
    
    def reset(self):
        self.total_entropy.reset()