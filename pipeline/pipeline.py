import logging
from tqdm import tqdm
import torch
from utils.process_data import get_model_generate
from data_loader.base_loader import BaseLoader
from data_processor.base_processor import BaseProcessor

class Pipeline:
    def __init__(self, model, tokenizer, model_config, data_loaders:list[BaseLoader], data_processors:list[BaseProcessor]):
        logging.info("Init Pipeline")
        self.model = model
        self.tokenizer = tokenizer 
        self.model_config = model_config
        self.data_loaders = data_loaders
        self.data_processors = data_processors
        self.min_input_token = 50
        self.max_input_token = 2000
        self.max_sample = 50

    def run(self):
        logging.info("Pipeline start")
        self.model.eval()
        with torch.no_grad():
            # data_loaders
            for data_loader in self.data_loaders:
                logging.info(f"Data loader {data_loader.name}")
                load_data = data_loader.load_data()
                split_words = data_loader.split_words()
                # init processor
                for data_processor in self.data_processors:
                    data_processor.set(data_loader.name)
                index = 0
                # data samples
                for data in load_data:
                    inputs = self.tokenizer(data, padding=False, return_tensors='pt')
                    num_input_token = inputs['input_ids'].shape[1]
                    if num_input_token < self.min_input_token or num_input_token > self.max_input_token:
                        logging.info(f"num_input_token {num_input_token} less than min_input_token {self.min_input_token} or greater than max_input_token {self.max_input_token}")
                        continue
                    # pre process
                    model_generate = get_model_generate(self.tokenizer,self.model,data,max_new_tokens=1,max_input_token=400,split_words=split_words)
                    index += 1 
                    # data_processors
                    for data_processor in self.data_processors:
                        if data_processor.process(index,data,model_generate,split_words=split_words):
                            logging.info(f"num_input_token {num_input_token} data processing...")
                    if index > self.max_sample:
                        break
            logging.info("Pipeline end")
