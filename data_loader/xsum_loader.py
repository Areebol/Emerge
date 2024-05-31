from data_loader.base_loader import BaseLoader
import logging

class XsumLoader(BaseLoader):
    def __init__(self):
        self.name = "xsum_examples"

    def load_data(self, data_len = 200):
        import os
        os.environ['HF_ENDPOINT']='https://hf-mirror.com'
        from datasets import load_dataset
        logging.info(f"Loading data {self.name} from HuggingFace Hub...")
        dataset = load_dataset("EdinburghNLP/xsum")
        xsum_examples = [f"Document:{document}.Summary:{summary}" for document,summary in zip(dataset['validation']['document'],dataset['validation']['summary'])]
        return xsum_examples[:data_len]
