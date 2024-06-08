import datasets
import numpy as np
from typing import List, Dict
from data_loader.base_loader import BaseLoader


class SentimentClassificationLoader(BaseLoader):
    def __init__(self) -> None:
        self.name = 'poem_sentiment_classification'
    
    def load_data(self, data_len: int=1000, seed: int=42, k_shots: int=15):
        dataset = datasets.load_dataset("poem_sentiment")
        label = {
            0:"negative",
            1:"positive",
            2:"no_impact",
            #3:"mixed", # there is no `mixed` on the test set
        }
        all_data = [line for line in dataset['train']] + \
           [line for line in dataset['test']]  + \
           [line for line in dataset['validation']]

        self.all_data = [{"verse_text": d['verse_text'].strip().replace("\\n", " "), 
                          "label": label[d['label']]} 
                          for d in all_data if d['label']!=3]
        examplers: List[Dict] = self.all_data[:100]
        np.random.seed(seed)
        np.random.shuffle(examplers)
        examplers: List[Dict] = examplers[:k_shots]

        test_data: List[Dict] = self.all_data[100:]
        demos: str = '\n'.join([f"Text: {d['verse_text']}\nSentiment: {d['label']}\n" for d in examplers])

        instruct: str = 'Classify the text into no_impact, negative, or positive\n'
        new_test_data: List[str] = [{"prompt": f"{instruct}\n{demos}\nText: {d['verse_text']}\nSentiment: ", "gt": d['label']} for d in test_data]

        return [d['prompt'] for d in new_test_data[:data_len]]
    
    def split_words(self):
        return ["Sentiment"]