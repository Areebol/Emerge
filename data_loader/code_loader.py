from data_loader.base_loader import BaseLoader
import logging

class CodeLoader(BaseLoader):
    def __init__(self):
        self.name = "Code_examples"

    def load_data(self, data_len = 200):
        import os
        os.environ['HF_ENDPOINT']='https://hf-mirror.com'
        from datasets import load_dataset
        logging.info(f"Loading data {self.name} from HuggingFace Hub...")
        dataset = load_dataset("sahil2801/CodeAlpaca-20k")
        code_examples = [f"Instruction:{instruction}\nCode:{code}" for instruction,code in zip(dataset['train']['instruction'],dataset['train']['output'])]
        return code_examples[:data_len]
