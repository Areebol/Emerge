import os
import json
import logging
from typing import List, Dict, Tuple
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import datasets
import numpy as np
from datasets.dataset_dict import DatasetDict

from data_loader.base_loader import BaseLoader


class ARCLoader(BaseLoader):
    def __init__(self, part: str='Challenge'):
        assert part in ['Challenge', 'Easy']
        self.name = f"arc_{part}_multi_choices_few_shot"
        self.dataset = datasets.load_dataset("allenai/ai2_arc", f"ARC-{part}")

    @staticmethod
    def map_hf_dataset_to_list(dataset: DatasetDict, split_name: str, add_newlines: bool=False) -> List[Tuple]:
        lines = []

        for datapoint in dataset[split_name]:
            answer_index = datapoint["answerKey"]
            choices_string = ""
            for i in range(len(datapoint["choices"]["label"])):
                if datapoint["choices"]["label"][i] == answer_index:
                    answer_string = f"({answer_index}) " + datapoint["choices"]["text"][i]
                
                if add_newlines: choices_string += "\n(" + datapoint["choices"]["label"][i] + ") " + datapoint["choices"]["text"][i]
                else:            choices_string += " (" + datapoint["choices"]["label"][i] + ") " + datapoint["choices"]["text"][i]
            lines.append((datapoint["question"] + choices_string, answer_string))

        return lines

    @staticmethod
    def preprocess(line: Tuple[str, str]) -> Dict:
        input_, output_ = line
        input_ = input_.strip().replace("\\n", " ")
        output_ = str(output_).split("\t")[0].strip()
        return {"input": input_, "output": output_}


    def load_data(self, data_len: int=1000, add_newlines: bool=True, seed: int=42, k_shot: int=15):
        assert k_shot < 100, 'Error for ARC data'
        logging.info(f"Loading data {self.name}()")
        all_data: List[Dict] = [self.preprocess(line) for line in self.map_hf_dataset_to_list(dataset=self.dataset, split_name='train', add_newlines=add_newlines)] + \
                               [self.preprocess(line) for line in self.map_hf_dataset_to_list(dataset=self.dataset, split_name='test', add_newlines=add_newlines)]  + \
                               [self.preprocess(line) for line in self.map_hf_dataset_to_list(dataset=self.dataset, split_name='validation', add_newlines=add_newlines)]
        
        examplers: List[Dict] = all_data[:100]
        np.random.seed(seed)
        np.random.shuffle(examplers)
        examplers: List[Dict] = examplers[:k_shot]

        test_data: List[Dict] = all_data[100:]
        demos: str = '\n'.join([f"Q: {d['input']}\nA: {d['output']}\n" for d in examplers])
        new_test_data: List[Dict] = [{"prompt": f"{demos}\nQ: {data['input']}", "gt": data['output']} for data in test_data]

        return [d['prompt'] for d in new_test_data[:data_len]]