from data_loader.base_loader import BaseLoader
import logging

class HC3HumanLoader(BaseLoader):
    def __init__(self):
        self.name = "hc3_human_examples"

    def load_data(self, data_len = 1000):
        import os
        os.environ['HF_ENDPOINT']='https://hf-mirror.com'
        from datasets import load_dataset
        logging.info(f"Loading data {self.name} from HuggingFace Hub...")
        dataset = load_dataset("Hello-SimpleAI/HC3","all")
        hc3_human_examples = [f"Question:{question} Human_answers:{human_answers[0]}".replace(" .", ".").replace(" ? ","?").replace("\n","") for question,human_answers in zip(dataset['train']['question'][0:data_len],dataset['train']['human_answers'][0:data_len])]

        return hc3_human_examples[:data_len]
